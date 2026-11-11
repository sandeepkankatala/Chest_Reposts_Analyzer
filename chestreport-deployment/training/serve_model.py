"""Small local inference server compatible with ChestReport's model client.
Start with: python -m uvicorn training.serve_model:app --host 127.0.0.1 --port 8001
Set CHESTREPORT_ADAPTER_PATH to a trained adapter directory, or leave unset for base.
"""
import os
import secrets
import threading
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from backend.prompts import MODEL_ID, PROMPT_VERSION

model=processor=None
lock=threading.Lock()
metadata={}

def authenticate(authorization=None, x_chestreport_key=None):
    key=os.getenv('MEDGEMMA_API_KEY', '')
    if not key or not (secrets.compare_digest(x_chestreport_key or '', key) or secrets.compare_digest(authorization or '', 'Bearer '+key)):
        raise HTTPException(401,'Invalid inference key.')

def adapter_manifest(adapter):
    if not adapter:
        return {}
    manifest=json.loads((Path(adapter)/'training_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('base_model')!=os.getenv('MEDGEMMA_MODEL',MODEL_ID) or manifest.get('prompt_version')!=PROMPT_VERSION:
        raise RuntimeError('Adapter base model or prompt version does not match the application.')
    if manifest.get('smoke_only'):
        raise RuntimeError('The two-step smoke adapter is not a completed training run.')
    if not manifest.get('base_revision'):
        raise RuntimeError('Adapter manifest must identify the exact base-model revision.')
    config=json.loads((Path(adapter)/'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path')!=manifest['base_model']:
        raise RuntimeError('Adapter configuration does not match the manifest base model.')
    return manifest

@asynccontextmanager
async def lifespan(app):
    global model,processor,metadata
    if len(os.getenv('MEDGEMMA_API_KEY',''))<24:
        raise RuntimeError('Set a random MEDGEMMA_API_KEY of at least 24 characters before serving.')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('This inference recipe requires a CUDA GPU.')
    name=os.getenv('MEDGEMMA_MODEL',MODEL_ID)
    adapter=os.getenv('CHESTREPORT_ADAPTER_PATH')
    manifest=adapter_manifest(adapter)
    revision=manifest.get('base_revision') or os.getenv('MEDGEMMA_REVISION')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=AutoProcessor.from_pretrained(name,revision=revision)
    model=AutoModelForImageTextToText.from_pretrained(name,revision=revision,device_map={'':torch.cuda.current_device()},torch_dtype=dtype,attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter)
    model.eval()
    metadata={'status':'ready','model':name,'adapter_loaded':bool(adapter),
        'adapter_version':manifest.get('adapter_version'),'base_revision':revision,
        'prompt_version':PROMPT_VERSION,'synthetic_starter':manifest.get('synthetic_starter',False),
        'clinical_validation':False}
    yield
    metadata={}

app=FastAPI(title='ChestReport local MedGemma inference',lifespan=lifespan)

@app.get('/health')
def health():
    if not metadata:raise HTTPException(503,'Model loading.')
    return {'status':'ready'}

@app.get('/v1/health')
def detailed_health(authorization:str|None=Header(default=None),x_chestreport_key:str|None=Header(default=None)):
    authenticate(authorization,x_chestreport_key)
    health()
    return metadata

class Message(BaseModel):
    role: str = Field(pattern=r'^(user|assistant)$')
    content: str = Field(min_length=1,max_length=30000)

class Completion(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1,max_length=1)
    max_tokens: int = Field(default=1800,ge=1,le=2048)
    temperature: float = Field(default=.1,ge=0,le=1)

@app.post('/v1/chat/completions')
def generate(data:Completion,authorization:str|None=Header(default=None),x_chestreport_key:str|None=Header(default=None)):
    authenticate(authorization,x_chestreport_key)
    health()
    if data.model!=metadata['model']:raise HTTPException(400,'Unknown model.')
    if data.messages[0].role!='user':raise HTTPException(422,'Send one evidence-grounded user turn.')
    import torch
    messages=[{'role':m.role,'content':[{'type':'text','text':m.content}]} for m in data.messages]
    if not lock.acquire(blocking=False):raise HTTPException(429,'GPU busy. Retry shortly.')
    try:
      with torch.inference_mode():
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
        prompt_tokens=inputs['input_ids'].shape[-1]
        if prompt_tokens+data.max_tokens>int(os.getenv('MEDGEMMA_CONTEXT_TOKENS','8192')):raise HTTPException(413,'Model context limit exceeded for this deployment.')
        outputs=model.generate(**inputs,max_new_tokens=data.max_tokens,do_sample=False)
        generated=outputs[0][prompt_tokens:]
        text=processor.decode(generated,skip_special_tokens=True)
        finish='length' if len(generated)>=data.max_tokens else 'stop'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(503,'GPU memory exhausted. Reduce input length or use a larger GPU.') from None
    finally:
        lock.release()
    return {'model':data.model,'choices':[{'message':{'role':'assistant','content':text},'finish_reason':finish}],
        'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':len(generated)}}
