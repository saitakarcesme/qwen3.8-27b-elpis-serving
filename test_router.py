import contextlib
import http.client
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location("elpis_router",Path(__file__).with_name("router.py"))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_affinity_and_busy_lane():
    p=m.Pool([1,2]);p.healthy=[True,True]
    a=p.acquire("job");p.release(a)
    assert p.acquire("job")==a
    b=p.acquire("job");assert b!=a
    with pytest.raises(TimeoutError):p.acquire("third",timeout=.01)
    p.release(b);p.release(a)
    for n in range(300):
        i=p.acquire(str(n));p.release(i)
    assert len(p.affinity)==256

def test_unhealthy_lane_is_not_selected():
    p=m.Pool([1,2]);p.set_health(1,True)
    assert p.acquire("x")==1
    p.release(1);p.set_health(1,False)
    with pytest.raises(ConnectionError):p.acquire("x")

@contextlib.contextmanager
def serving(server):
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    try:yield server.server_address[1]
    finally:server.shutdown();server.server_close();t.join()

class Backend(BaseHTTPRequestHandler):
    calls=[]
    def log_message(self,*args):pass
    def do_POST(self):
        assert self.headers["Authorization"]=="Bearer local-test-key"
        body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append(body)
        raw=json.dumps({"received_model":body["model"]}).encode()
        self.send_response(200);self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)

def test_http_forwarding_and_context_metadata(tmp_path):
    key=tmp_path/"key";key.write_text("local-test-key");Backend.calls=[]
    with serving(ThreadingHTTPServer(("127.0.0.1",0),Backend)) as port:
        router=m.Server({"ports":[port],"port":0,"api_key_file":str(key),"usage_log":None});router.pool.set_health(0,True)
        with serving(router) as front:
            c=http.client.HTTPConnection("127.0.0.1",front,timeout=3)
            c.request("GET","/v1/models");r=c.getresponse();assert r.status==200
            assert json.load(r)["data"][0]["max_model_len"]==262144
            c.request("POST","/v1/chat/completions",json.dumps({"model":"qwen3.8:27b","messages":[]}),{"Content-Type":"application/json"})
            r=c.getresponse();assert r.status==200;assert json.load(r)["received_model"]=="qwen3.8-27b";c.close()
            assert len(Backend.calls)==1 and router.pool.busy==[False]

def test_no_post_replay_when_backend_unreachable(tmp_path):
    key=tmp_path/"key";key.write_text("local-test-key")
    router=m.Server({"ports":[1,2],"port":0,"api_key_file":str(key),"usage_log":None});router.pool.healthy=[True,True]
    with serving(router) as front:
        c=http.client.HTTPConnection("127.0.0.1",front,timeout=3)
        c.request("POST","/v1/chat/completions",json.dumps({"model":"qwen3.8-27b"}))
        r=c.getresponse();assert r.status==503;assert "not replayed" in json.load(r)["error"];c.close()
        assert router.pool.completed==[1,0] and router.pool.busy==[False,False]


@pytest.mark.parametrize("path,model",[("/api/chat","qwen2.5vl:3b"),("/api/generate","qwen2.5vl:3b"),("/v1/chat/completions","qwen2.5vl:3b"),("/v1/chat/completions","qwen3.8:27b"),("/v1/chat/completions","qwen3.8-27b"),("/v1/chat/completions","qwen3.8-27b-sampled")])
def test_vision_reserves_secondary_and_restarts_it(tmp_path,monkeypatch,path,model):
    commands=[]
    monkeypatch.setattr(m.subprocess,"run",lambda args,**kwargs:commands.append(args))
    class Vision(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            raw=b'{"models":[]}'
            self.send_response(200);self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["keep_alive"]==0
            assert body["model"]==("qwen27-vision" if model in m.TEXT_MODELS or model==m.SAMPLED_MODEL else model)
            if self.path=="/v1/chat/completions":
                assert body["messages"][0]["content"][0]["image_url"]["url"]=="data:image/jpeg;base64,fixture"
            raw=json.dumps({"message":{"content":"vision ok"}}).encode()
            self.send_response(200);self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
    key=tmp_path/"key";key.write_text("local-test-key")
    with serving(ThreadingHTTPServer(("127.0.0.1",0),Vision)) as vision_port:
        router=m.Server({"ports":[1,2],"port":0,"api_key_file":str(key),"usage_log":None,"vision_port":vision_port,
                         "multimodal_model":"qwen27-vision","vision_model":"qwen2.5vl:3b","vision_container":"llmrig-elpis-lane1","vision_lane":1})
        router.pool.healthy=[True,True];router.pool.busy[0]=True
        with serving(router) as front:
            c=http.client.HTTPConnection("127.0.0.1",front,timeout=3)
            c.request("POST",path,json.dumps({"model":model,"stream":False,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,fixture"}}]}]}))
            r=c.getresponse();assert r.status==200;assert json.load(r)["message"]["content"]=="vision ok";c.close()
            end=time.monotonic()+2
            while router.pool.busy[1] and time.monotonic()<end:time.sleep(.01)
            assert router.pool.busy==[True,False]
            assert commands==[["docker","stop","--time","45","llmrig-elpis-lane1"],["docker","start","llmrig-elpis-lane1"]]


def test_stream_is_forwarded_before_backend_completion(tmp_path):
    consumed=threading.Event()
    class Streaming(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Transfer-Encoding","chunked")
            self.end_headers()
            first=b"data: first\n\n"
            self.wfile.write(f"{len(first):x}\r\n".encode()+first+b"\r\n")
            self.wfile.flush()
            assert consumed.wait(3),"Router buffered the first SSE frame"
            last=b"data: [DONE]\n\n"
            self.wfile.write(f"{len(last):x}\r\n".encode()+last+b"\r\n0\r\n\r\n")
            self.wfile.flush()
    key=tmp_path/"key";key.write_text("local-test-key")
    with serving(ThreadingHTTPServer(("127.0.0.1",0),Streaming)) as backend:
        router=m.Server({"ports":[backend],"port":0,"api_key_file":str(key),"usage_log":None})
        router.pool.set_health(0,True)
        with serving(router) as front:
            c=http.client.HTTPConnection("127.0.0.1",front,timeout=3)
            c.request("POST","/v1/chat/completions",json.dumps({"model":"qwen3.8-27b","stream":True}))
            response=c.getresponse()
            assert response.status==200
            assert response.readline()==b"data: first\n"
            consumed.set()
            assert b"data: [DONE]" in response.read()
            c.close()


def test_pdf_lock_excludes_secondary_but_primary_still_serves(tmp_path):
    import fcntl
    lock=tmp_path/'gpu.lock'
    p=m.Pool([1,2],str(lock));p.healthy=[True,True]
    with lock.open('a') as pdf:
        fcntl.flock(pdf,fcntl.LOCK_EX)
        assert p.acquire('text')==0
        with pytest.raises(TimeoutError):p.acquire('second',timeout=.03)
        with pytest.raises(TimeoutError):p.reserve(1,timeout=.03)
        p.release(0)
    assert p.reserve(1)==1
    with lock.open('a') as pdf:
        with pytest.raises(BlockingIOError):fcntl.flock(pdf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.release(1)
    with lock.open('a') as pdf:fcntl.flock(pdf,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_pdf_multipart_and_binary_download_passthrough(tmp_path):
    upload=b'--fixture\r\nContent-Disposition: form-data; name="file"; filename="test.pdf"\r\n\r\n%PDF-test\x00\xff\r\n--fixture--\r\n'
    class PDF(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            assert self.headers['Content-Type']=='multipart/form-data; boundary=fixture'
            assert self.rfile.read(int(self.headers['Content-Length']))==upload
            raw=b'{"id":"fixture","status":"queued"}'
            self.send_response(202);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            raw=b'PK\x00\xffbinary-zip'
            self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename=test.zip');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    key=tmp_path/'key';key.write_text('local-test-key')
    with serving(ThreadingHTTPServer(('127.0.0.1',0),PDF)) as backend:
        router=m.Server({'ports':[1,2],'port':0,'api_key_file':str(key),'usage_log':None,'pdf_port':backend})
        with serving(router) as front:
            c=http.client.HTTPConnection('127.0.0.1',front,timeout=3)
            c.request('POST','/v1/documents',upload,{'Content-Type':'multipart/form-data; boundary=fixture'})
            r=c.getresponse();assert r.status==202;assert json.load(r)['id']=='fixture';c.close()
            c=http.client.HTTPConnection('127.0.0.1',front,timeout=3)
            c.request('GET','/v1/documents/fixture/download');r=c.getresponse()
            assert r.status==200 and r.getheader('Content-Disposition')=='attachment; filename=test.zip'
            assert r.read()==b'PK\x00\xffbinary-zip';c.close()
            assert router.pool.completed==[0,0]


def test_props_reports_real_memory_limits_and_external_gpu_lease(tmp_path,monkeypatch):
    import fcntl
    calls=[]
    def memory():
        calls.append(1)
        return [{'index':0,'total_mib':24576,'used_mib':23000,'free_mib':1100}]
    monkeypatch.setattr(m,'gpu_memory',memory)
    key=tmp_path/'key';key.write_text('local-test-key');lock=tmp_path/'gpu.lock'
    router=m.Server({'ports':[1,2],'port':0,'api_key_file':str(key),'usage_log':None,'media_lock':str(lock)})
    router.pool.healthy=[True,True]
    with serving(router) as front, lock.open('a') as pdf:
        fcntl.flock(pdf,fcntl.LOCK_EX)
        c=http.client.HTTPConnection('127.0.0.1',front,timeout=3)
        c.request('GET','/props');r=c.getresponse();assert r.status==200;data=json.load(r)
        assert data['gpus'][0]['free_mib']==1100
        assert data['slots'][0]['available'] and not data['slots'][1]['available']
        assert all(slot['context_limit']==262144 and slot['context_tokens_in_use'] is None for slot in data['slots'])
        assert data['text_sampling']['presence_penalty']==0
        c.request('GET','/props');r=c.getresponse();r.read();c.close();assert len(calls)==1
    router.pool.telemetry_at=float('-inf')
    def broken():raise OSError('no GPU telemetry')
    monkeypatch.setattr(m,'gpu_memory',broken)
    data=m.props(router);assert data['gpus'] is None and data['gpu_memory_error']=='OSError'

def sse(obj):return b"data: "+json.dumps(obj).encode()+b"\n\n"

def test_usage_tap_hides_unrequested_stream_usage_and_records(tmp_path):
    payload={"model":"qwen3.8-27b","stream":True,"messages":[]}
    tap=m.UsageTap("/v1/chat/completions",payload,{"X-LLMRig-Affinity":"job-1"})
    assert payload["stream_options"]=={"include_usage":True}
    usage={"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}
    stream=sse({"choices":[{"delta":{"content":"hi"}}],"usage":None})+sse({"choices":[],"usage":usage})+b"data: [DONE]\n\n"
    out=b"".join(tap.feed(stream[i:i+7]) for i in range(0,len(stream),7))+tap.flush()
    assert b'"choices": []' not in out and out.endswith(b"data: [DONE]\n\n") and b'"hi"' in out
    tap.status=200;tap.lane=1;log=tmp_path/"usage.jsonl";tap.record(str(log))
    row=json.loads(log.read_text())
    assert (row["prompt_tokens"],row["completion_tokens"],row["cached_tokens"],row["lane"],row["session"])==(10,5,4,1,"job-1")

def test_usage_tap_keeps_requested_usage_frame():
    payload={"stream":True,"stream_options":{"include_usage":True}}
    tap=m.UsageTap("/v1/chat/completions",payload,{})
    frame=sse({"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}})
    assert tap.feed(frame)==frame and tap.usage["completion_tokens"]==2

def test_usage_tap_non_stream_and_native():
    tap=m.UsageTap("/v1/chat/completions",{"stream":False},{})
    body=json.dumps({"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}).encode()
    assert tap.feed(body[:5])+tap.feed(body[5:])==body and tap.flush()==b"" and tap.usage["prompt_tokens"]==3
    tap=m.UsageTap("/api/chat",{"model":"x"},{})
    data=b'{"done":false}\n{"done":true,"prompt_eval_count":7,"eval_count":8,"model":"x"}\n'
    assert tap.feed(data)==data and tap.flush()==b"" and tap.usage=={"prompt_tokens":7,"completion_tokens":8}

def test_conversation_key_is_stable_across_growing_history():
    first=[{"role":"system","content":"s"},{"role":"user","content":"hi"}]
    grown=first+[{"role":"assistant","content":"a"},{"role":"user","content":"more"}]
    k=m.conversation_key({"model":"qwen3.8-27b","messages":first})
    assert k.startswith("auto:") and k==m.conversation_key({"model":"qwen3.8-27b","messages":grown})
    assert k!=m.conversation_key({"model":"qwen3.8-27b","messages":[{"role":"system","content":"s"},{"role":"user","content":"other"}]})
    assert m.conversation_key({"model":"x"})==""
    assert m.conversation_key({"model":"x","prompt":"abc"}).startswith("auto:")

def test_sequential_requests_of_one_conversation_stay_on_one_lane():
    p=m.Pool([1,2]);p.healthy=[True,True]
    key=m.conversation_key({"model":"x","messages":[{"role":"user","content":"hi"}]})
    lanes=set()
    for _ in range(6):
        i=p.acquire(key);lanes.add(i);p.release(i)
    assert len(lanes)==1
