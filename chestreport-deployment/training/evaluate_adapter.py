"""Compare base and adapter on exactly the same held-out synthetic prompts.
Scores are format/evidence checks, not evidence of clinical safety or correctness.
"""
import argparse
import json
import re
import time
from pathlib import Path
from training.serve_model import adapter_manifest

def score_output(text, row):
    score={'json_valid':False,'schema_valid':False,'citations_valid':False,'finding_status_exact':None,'abstention_correct':None,'summary_exact':None}
    if row['task'].startswith('analysis'):
        score.update(finding_status_exact=False,summary_exact=False)
    else:
        score['abstention_correct']=False
    try:
        got=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip()))
        expected=json.loads(row['messages'][1]['content'])
        score['json_valid']=True
        sources={s['id']:s['text'] for s in row['sources']}
        if row['task'].startswith('analysis'):
            claims=got['findings']+got['summary']
            score['schema_valid']=bool(got['summary']) and all(isinstance(x.get('text'),str) for x in got['summary']) and all(isinstance(x.get('name'),str) and isinstance(x.get('explanation'),str) and x.get('status') in ('present','absent','uncertain') for x in got['findings'])
            facts=lambda obj:sorted((x['name'].lower(),x['status'],x['evidence']) for x in obj['findings'])
            score['finding_status_exact']=facts(got)==facts(expected)
            score['summary_exact']=[x['text'] for x in got['summary']]==[x['text'] for x in expected['summary']]
        else:
            claims=got['citations']
            score['schema_valid']=isinstance(got['answer'],str) and isinstance(got['abstained'],bool) and (got['abstained'] or bool(claims))
            score['abstention_correct']=got['abstained']==expected['abstained']
        score['citations_valid']=score['schema_valid'] and all(isinstance(c.get('evidence'),str) and bool(c['evidence']) and c['evidence'] in sources.get(c.get('source_id'),'') for c in claims)
    except (ValueError,KeyError,TypeError,AttributeError):
        pass
    return score

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--test',required=True)
    parser.add_argument('--output',default='training/output/comparison.json')
    parser.add_argument('--limit',type=int,default=16,help='Use 0 for the full held-out split.')
    args=parser.parse_args()
    manifest=adapter_manifest(args.adapter)
    rows=[json.loads(s) for s in Path(args.test).read_text(encoding='utf-8').splitlines() if s.strip()]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No test examples.')
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(manifest['base_model'],revision=manifest['base_revision'])
    model=AutoModelForImageTextToText.from_pretrained(manifest['base_model'],revision=manifest['base_revision'],device_map={'':0},torch_dtype=dtype,attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    results={}
    # Generate base first, then load the trained adapter into the same model.
    for label in ('base','adapter'):
        if label=='adapter':model=PeftModel.from_pretrained(model,args.adapter)
        model.eval()
        predictions=[]
        for row in rows:
            messages=[{'role':'user','content':[{'type':'text','text':row['messages'][0]['content']}]}]
            inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=768,do_sample=False)
            torch.cuda.synchronize()
            text=processor.decode(outputs[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
            predictions.append({'task':row['task'],'group':row['patient_id'],'seconds':time.perf_counter()-start,'output':text,**score_output(text,row)})
        summary={}
        for key in ('json_valid','schema_valid','citations_valid','finding_status_exact','abstention_correct','summary_exact'):
            values=[p[key] for p in predictions if p[key] is not None]
            summary[key]={'rate':sum(values)/len(values) if values else None,'count':len(values)}
        summary['mean_seconds']=sum(p['seconds'] for p in predictions)/len(predictions)
        results[label]={'summary':summary,'predictions':predictions}
    results['limitation']='Shared synthetic templates; quotation checks and exact target matches do not establish clinical correctness. Manually review outputs.'
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({k:v['summary'] for k,v in results.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':main()
