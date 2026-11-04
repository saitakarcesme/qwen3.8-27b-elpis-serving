"""Read-only deployment check. Never prints the API key or job contents."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"router.json").read_text())
auth={"Authorization":"Bearer "+Path(cfg["api_key_file"]).read_text().strip()}
report={"checks":[]}
def check(name,passed,details=None):
 report["checks"].append({"name":name,"passed":bool(passed),"details":details})
def get(port,path,headers=None):
 req=urllib.request.Request(f"http://127.0.0.1:{port}{path}",headers=headers or {})
 with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)
for port in [cfg["port"],*cfg.get("alias_ports",[])]:
 try:
  h=get(port,"/health")
  check(f"router:{port}",h["context"]==262144 and all(x["healthy"] for x in h["lanes"]),h)
 except Exception as error:check(f"router:{port}",False,type(error).__name__)
for port in cfg["ports"]:
 try:
  models=get(port,"/v1/models",auth)["data"]
  check(f"engine:{port}",any(x["id"]=="qwen3.8-27b" and x["max_model_len"]==262144 for x in models),models)
 except Exception as error:check(f"engine:{port}",False,type(error).__name__)
for unit in ["llmrig-elpis","contest-forge-worker","contest-forge-scanner"]:
 p=subprocess.run(["systemctl","--user","is-active",unit],capture_output=True,text=True)
 check(unit,p.returncode==0,p.stdout.strip())
p=subprocess.run(["systemctl","--user","is-active","llmrig-homelab"],capture_output=True,text=True)
check("previous-engine-stopped",p.returncode!=0,p.stdout.strip())
for container in ["llmrig-elpis-lane0","llmrig-elpis-lane1"]:
 p=subprocess.run(["docker","inspect",container,"--format","{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}"],capture_output=True,text=True)
 check(container,p.returncode==0 and p.stdout.strip()=="running false 0",p.stdout.strip())
sys.path.insert(0,"/home/isa/Documents/ChatGPT/Contest")
from contest_forge.config import load
c=load(Path("/home/isa/Documents/ChatGPT/Contest"))
check("contest-settings",c["model"]["base_url"]=="http://127.0.0.1:11446/v1" and
 c["model"].get("temperature")==0 and c["loop"].get("max_parallel_jobs")==2 and
 c["media"]["vision_url"]=="http://127.0.0.1:11446",
 {"model":c["model"]["builder"],"timeout":c["model"]["timeout_seconds"],
  "output_budget":c["model"]["max_tokens"],"parallel_jobs":c["loop"].get("max_parallel_jobs")})
report["passed"]=all(x["passed"] for x in report["checks"])
print(json.dumps(report,indent=2))
raise SystemExit(0 if report["passed"] else 1)
