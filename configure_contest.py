"""Preserve unrelated settings and update only the qualified local inference recipe."""
import json
import re
import subprocess
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path("/home/isa/Documents/ChatGPT/Contest")
UPDATES = {
    "model": {
        "base_url": "http://127.0.0.1:11446/v1",
        "builder": "qwen3.8-27b",
        "verifier": "qwen3.8-27b",
        "temperature": 0,
        "timeout_seconds": 7200,
        "max_tokens": 8192,
        "reasoning_effort": "low",
    },
    "loop": {"max_parallel_jobs": 2},
    "media": {"vision_url": "http://127.0.0.1:11446", "timeout_seconds": 7200},
}

def update(text, section, values):
    header = re.search(r"(?m)^\[" + re.escape(section) + r"\]\s*$", text)
    if header is None:
        return text.rstrip() + "\n\n[" + section + "]\n" + "".join(
            k + " = " + json.dumps(v) + "\n" for k, v in values.items()
        )
    next_header = re.search(r"(?m)^\[", text[header.end():])
    end = header.end() + next_header.start() if next_header else len(text)
    block = text[header.end():end]
    for key, value in values.items():
        line = key + " = " + json.dumps(value)
        pattern = r"(?m)^" + re.escape(key) + r"\s*=.*$"
        if re.search(pattern, block):
            block = re.sub(pattern, lambda _: line, block)
        else:
            block = block.rstrip() + "\n" + line + "\n"
    return text[:header.end()] + block + "\n" + text[end:]

if __name__ == "__main__":
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                       "contest-forge-worker.service"]).returncode == 0:
        raise SystemExit("Stop the worker before updating its recipe.")
    with urllib.request.urlopen("http://127.0.0.1:11446/health", timeout=10) as response:
        health = json.load(response)
    assert health["context"] == 262144 and len(health["lanes"]) == 2
    assert all(lane["healthy"] for lane in health["lanes"])
    path = ROOT / "config.local.toml"
    text = path.read_text() if path.exists() else ""
    before = tomllib.loads(text)
    for section, values in UPDATES.items():
        text = update(text, section, values)
    after = tomllib.loads(text)
    for section, values in before.items():
        for key, value in values.items():
            if key not in UPDATES.get(section, {}):
                assert after[section][key] == value
    for section, values in UPDATES.items():
        assert all(after[section][key] == value for key, value in values.items())
    pending = path.with_suffix(".toml.elpis-pending")
    pending.write_text(text); pending.chmod(0o600); pending.replace(path)
    print("Updated only model, concurrency and local vision routing settings.")
