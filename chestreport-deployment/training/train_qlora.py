"""Text-only QLoRA adaptation of MedGemma. Requires CUDA and approved model access.
Not run as part of the web application. The vision encoder remains frozen.
"""
import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path
from backend.prompts import PROMPT_VERSION
from training.prepare_synthetic import PROVENANCE

def read_reviewed(path, allow_synthetic=False):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(row.get('reviewed') is not True and not (allow_synthetic and row.get('provenance') == PROVENANCE and row.get('patient_id', '').startswith('synthetic-')) for row in rows):
        raise ValueError('Use non-empty, reviewed datasets (reviewed: true on every row).')
    for row in rows:
        messages=row.get('messages',[])
        if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
            raise ValueError('Each example must contain one user message and one assistant target.')
        if not row.get('patient_id') or not all(isinstance(m['content'],str) and m['content'].strip() for m in messages):
            raise ValueError('Each row needs a grouping ID and non-empty text messages.')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',required=True);parser.add_argument('--validation',required=True)
    parser.add_argument('--output',default='training/output/chestreport-adapter')
    parser.add_argument('--model',default='google/medgemma-1.5-4b-it')
    parser.add_argument('--epochs',type=int,default=2);parser.add_argument('--max-length',type=int,default=1024)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--max-steps',type=int,default=-1,help='Use 2 for a memory smoke test; -1 for full training.')
    parser.add_argument('--allow-synthetic-starter',action='store_true',help='Explicitly permit the unreviewed authored starter; does not mark it reviewed.')
    parser.add_argument('--revision',default=None,help='Resolved base-model commit for reproducibility.')
    args=parser.parse_args()
    train=read_reviewed(args.train,args.allow_synthetic_starter);validation=read_reviewed(args.validation,args.allow_synthetic_starter)
    if {r['patient_id'] for r in train}&{r['patient_id'] for r in validation}:
        raise ValueError('Patient/group overlap between train and validation.')
    if {r['messages'][0]['content'] for r in train}&{r['messages'][0]['content'] for r in validation}:
        raise ValueError('Duplicate prompts across splits.')
    import torch
    from datasets import Dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for this QLoRA recipe.')
    set_seed(42)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(args.model,revision=args.revision)
    tokenizer=processor.tokenizer
    model=AutoModelForImageTextToText.from_pretrained(args.model,revision=args.revision,device_map={'':torch.cuda.current_device()},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),torch_dtype=dtype,attn_implementation='sdpa')
    base_revision = getattr(model.config, '_commit_hash', None) or args.revision
    model=prepare_model_for_kbit_training(model)
    # Target only language layers; do not adapt similarly named vision projections.
    targets=[name for name,module in model.named_modules() if 'language_model' in name and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj'}]
    if not targets:raise RuntimeError('Language projection names changed; inspect the model before training.')
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=2*args.rank,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    def encode(rows):
        examples=[]
        for row in rows:
            messages=[{'role':m['role'],'content':[{'type':'text','text':m['content']}]} for m in row['messages']]
            prompt=tokenizer(processor.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']
            full=tokenizer(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False),add_special_tokens=False)['input_ids']
            if len(full)>args.max_length:continue
            if full[:len(prompt)]!=prompt:raise ValueError('Chat-template prefix mismatch; refusing incorrect loss masking.')
            examples.append({'input_ids':full,'attention_mask':[1]*len(full),'labels':[-100]*len(prompt)+full[len(prompt):]})
        if not examples:raise ValueError('No examples fit the token budget.')
        print(f'Using {len(examples)} / {len(rows)} examples within token budget.')
        return Dataset.from_list(examples)
    def collate(batch):
        length=max(len(row['input_ids']) for row in batch)
        return {key:torch.tensor([row[key]+[tokenizer.pad_token_id if key=='input_ids' else -100 if key=='labels' else 0]*(length-len(row[key])) for row in batch]) for key in ('input_ids','attention_mask','labels')}
    model.print_trainable_parameters()
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=args.output,num_train_epochs=args.epochs,max_steps=args.max_steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},learning_rate=1e-4,warmup_ratio=.05,logging_steps=1 if args.max_steps>0 else 10,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,save_total_limit=1,optim='paged_adamw_8bit',bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,report_to='none',seed=42),train_dataset=encode(train),eval_dataset=encode(validation),data_collator=collate)
    trainer.train();metrics=trainer.evaluate();trainer.save_model(args.output);processor.save_pretrained(args.output)
    Path(args.output,'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    manifest={'base_model':args.model,'base_revision':base_revision,'prompt_version':PROMPT_VERSION,
        'adapter_version':Path(args.output).name,'synthetic_starter':any(r.get('provenance')==PROVENANCE for r in train),
        'clinical_validation':False,'smoke_only':args.max_steps>0,'parameters':vars(args),
        'train_sha256':hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        'validation_sha256':hashlib.sha256(Path(args.validation).read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(),'peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,
        'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','bitsandbytes','datasets')}}
    Path(args.output,'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Adapter saved. Run held-out clinical evaluation before connecting it to the application.')

if __name__=='__main__':main()
