import array
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

root=Path(__file__).resolve().parents[1]
project=root/'chestreport'
sys.path[:0]=[str(project),str(project/'.packages')]
from training.evaluate_adapter import score_output
from training.serve_model import adapter_manifest

folder=root/'chestreport-deployment'
archive=Path.home()/'Downloads/chestreport-deployment.zip'
result={}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    files=[n for n in z.namelist() if not n.endswith('/')]
    assert all((folder/n).read_bytes()==z.read(n) for n in files)
    result['archive']={'crc_ok':True,'extracted_files_match':len(files),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
manifest=adapter_manifest(str(folder/'adapter'))
result['manifest_valid']=True
splits={name:[json.loads(line) for line in (folder/'synthetic'/f'{name}.jsonl').read_text().splitlines()] for name in ('train','validation','test')}
for name in ('train','validation'):
    assert hashlib.sha256((folder/'synthetic'/f'{name}.jsonl').read_bytes()).hexdigest()==manifest[name+'_sha256']
groups={name:{r['patient_id'] for r in rows} for name,rows in splits.items()}
prompts={name:{r['messages'][0]['content'] for r in rows} for name,rows in splits.items()}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not groups[a]&groups[b]
    assert not prompts[a]&prompts[b]
result['dataset']={'hashes_match':True,'counts':{k:len(v) for k,v in splits.items()},'group_counts':{k:len(v) for k,v in groups.items()},'no_cross_split_groups_or_prompts':True}
raw=(folder/'adapter/adapter_model.safetensors').read_bytes()
length=struct.unpack('<Q',raw[:8])[0]
header=json.loads(raw[8:8+length])
data=raw[8+length:]
tensors={k:v for k,v in header.items() if k!='__metadata__'}
intervals=[]
parameters=0
b_nonzero=0
for name,tensor in tensors.items():
    assert 'language_model' in name and 'vision' not in name
    assert tensor['dtype']=='F32',tensor['dtype']
    start,end=tensor['data_offsets']
    size=math.prod(tensor['shape'])
    assert end-start==4*size and 0<=start<end<=len(data)
    assert 8 in tensor['shape']
    values=array.array('f',data[start:end])
    if sys.byteorder!='little':values.byteswap()
    assert all(math.isfinite(v) for v in values)
    if '.lora_B.' in name:
        assert any(v!=0 for v in values)
        b_nonzero+=1
    intervals.append((start,end));parameters+=size
intervals.sort()
assert intervals[0][0]==0 and intervals[-1][1]==len(data)
assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
result['weights']={'tensor_count':len(tensors),'parameters':parameters,'nonzero_B_matrices':b_nonzero,'finite_and_layout_valid':True,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
comparison=json.loads((folder/'comparison.json').read_text())
lookup={(r['patient_id'],r['task']):r for r in splits['test']}
for label in ('base','adapter'):
    predictions=comparison[label]['predictions']
    assert len({(p['group'],p['task']) for p in predictions})==len(predictions)
    rescored=[]; diagnostic=[]
    for pred in predictions:
        row=lookup[pred['group'],pred['task']]
        scores=score_output(pred['output'],row)
        assert all(scores[k]==pred[k] for k in scores)
        rescored.append(scores)
        # Diagnostic only: do not modify the official strict app-contract scores.
        text=pred['output'].split('<unused95>',1)[-1]
        diagnostic.append(score_output(text,row))
    for key,metric in comparison[label]['summary'].items():
        if key=='mean_seconds':continue
        values=[s[key] for s in rescored if s[key] is not None]
        assert metric['count']==len(values) and metric['rate']==sum(values)/len(values)
    task_times=collections.defaultdict(list)
    for p in predictions:task_times[p['task']].append(p['seconds'])
    result[label]={'saved_metrics_reproduced':True,'summary':comparison[label]['summary'],'evaluated_reports':len({p['group'] for p in predictions}),'mean_seconds_by_task':{k:sum(v)/len(v) for k,v in task_times.items()},'diagnostic_final_segment_json_valid':sum(s['json_valid'] for s in diagnostic)}
result['source_matches']={name:(folder/name).read_bytes()==(project/name).read_bytes() for name in ('backend/prompts.py','training/serve_model.py','training/evaluate_adapter.py','training/train_qlora.py')}
result['limits']=['Inspected downloaded artifacts and rescored saved outputs; did not rerun GPU inference.','Only four synthetic report groups were evaluated, with sentence templates shared with training.','Base strict JSON failure is partly output-format mismatch; it is not zero medical accuracy.','Deployment image torch 2.6/CUDA 12.4 differs from recorded training torch 2.14.1/CUDA 13.0; verify adapter loading on the deployment GPU.']
target=folder/'verification_results.json'
target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
