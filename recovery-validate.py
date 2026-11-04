import base64, concurrent.futures, json, struct, subprocess, threading, time, urllib.request, zlib
from pathlib import Path
OUT=Path('/home/isa/Projects/llmrig/recipes/elpis-dual-262k/recovery-validation-20261030.json')
BASE='http://127.0.0.1:11446'
samples=[]; results=[]; stop=threading.Event()
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=8) as r:return json.load(r)
def sample():
    while not stop.is_set():
        try:samples.append(get('/props'))
        except Exception as e:samples.append({'error':type(e).__name__,'sampled_at':time.time()})
        stop.wait(2)
def save():OUT.write_text(json.dumps({'results':results,'samples':samples},indent=2))
def post(label,messages,affinity='recovery-validation-gpu0'):
    body={'model':'qwen3.8-27b','messages':messages,'max_tokens':64,'temperature':0,'stream':False}
    start=time.monotonic()
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-LLMRig-Affinity':affinity})
    with urllib.request.urlopen(req,timeout=240) as r:
        data=json.load(r)
        result={'label':label,'status':r.status,'seconds':round(time.monotonic()-start,3),'lane':r.headers.get('X-LLMRig-Lane'),'usage':data.get('usage'),'choices':data.get('choices')}
    results.append(result);save();print(json.dumps(result),flush=True)
    return result
def ready():
    end=time.monotonic()+240
    while time.monotonic()<end:
        p=get('/props')
        if all(s['available'] for s in p['slots']):return p
        time.sleep(2)
    raise RuntimeError('Lanes did not become idle and healthy; no additional test submitted')
def text(words):return [{'role':'user','content':('alpha beta gamma delta. '*words)+'\nReply with OK only.'}]
def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',128,128,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*128)*128))+chunk(b'IEND',b'')
vision=[{'role':'user','content':[{'type':'text','text':'What color is this square? Answer in one word.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]}]
threading.Thread(target=sample,daemon=True).start()
try:
    ready();post('short-text',text(0))
    ready();post('roughly-20k-text',text(4000))
    p=ready()
    if p['gpus'][0]['free_mib']<700:raise RuntimeError('GPU0 memory guard: below 700 MiB')
    post('vision-sequential',vision)
    ready();post('text-after-vision',text(0))
    p=ready()
    if p['gpus'][0]['free_mib']<700:raise RuntimeError('GPU0 memory guard before overlap')
    with concurrent.futures.ThreadPoolExecutor() as pool:
        f=pool.submit(post,'vision-overlap',vision)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and not f.done():
            p=get('/props')
            if p['slots'][1]['gpu_lease_held'] and p['slots'][1]['busy']:
                if p['slots'][0]['available'] and p['gpus'][0]['free_mib']>=700:
                    post('text-during-vision-20k',text(4000))
                break
            time.sleep(.5)
        f.result()
    ready();post('final-text',text(0));ready()
except Exception as e:
    results.append({'error':str(e),'type':type(e).__name__});print(json.dumps(results[-1]),flush=True)
finally:
    stop.set();save()
