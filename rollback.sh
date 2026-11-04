#!/usr/bin/env bash
# Restore the previous engine/configuration, preserving the Elpis files.
set -euo pipefail
recipe=/home/isa/Projects/llmrig/recipes/elpis-dual-262k
research=/home/isa/Experiments/elpis-dual-262k-20261029
contest=/home/isa/Documents/ChatGPT/Contest
stamp=$(date -u +%Y%m%dT%H%M%SZ)
systemctl --user stop contest-forge-scanner.service contest-forge-worker.service
systemctl --user disable --now llmrig-elpis.service
if [[ -f "$contest/config.local.toml" ]]; then
  mv "$contest/config.local.toml" "$research/backups/config.local.elpis-disabled-$stamp.toml"
fi
if [[ -f "$research/backups/config.local.toml" ]]; then
  cp "$research/backups/config.local.toml" "$contest/config.local.toml"
  chmod 600 "$contest/config.local.toml"
fi
dropin=/home/isa/.config/systemd/user/contest-forge-worker.service.d/60-elpis.conf
if [[ -f "$dropin" ]]; then
  mv "$dropin" "$research/backups/contest-worker-elpis-disabled-$stamp.conf"
fi
systemctl --user daemon-reload
systemctl --user enable --now llmrig-homelab.service
python - <<'PY'
import time,urllib.request
for attempt in range(300):
 try:
  with urllib.request.urlopen('http://127.0.0.1:11446/health',timeout=2) as r:
   if r.status==200:break
 except OSError:pass
 time.sleep(1)
else:raise SystemExit('Previous engine did not become healthy; clients remain stopped.')
PY
systemctl --user start contest-forge-worker.service contest-forge-scanner.service
