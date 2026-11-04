#!/usr/bin/env bash
set -euo pipefail
root=/home/isa/Projects/llmrig/recipes/elpis-dual-262k
if systemctl --user is-active --quiet llmrig-homelab.service; then
  echo "Original Qwen service is active; stop its clients and service first." >&2; exit 1
fi
/usr/bin/python "$root/media_session.py" "$root/router.json"
expected=sha256:2446a45495af44611a5a3c4ee1986a1f79a6acb29d04619629484d6d4527486d
actual=$(docker image inspect --format '{{.Id}}' llmrig-elpis:20261029)
test "$actual" = "$expected"
docker run --rm --network none --user 1000:1000 --read-only --mount type=bind,source=/home/isa/models/elpis-262k,target=/models,readonly --entrypoint /opt/venv/bin/python "$expected" -B /model-preparation/verify-models.py --target /models/qwen38-27b-exl3 --draft /models/dflash2-exl3 --representation exl3
docker start homelab-ollama > /dev/null
docker compose --project-name llmrig-elpis-dual --file "$root/compose.yaml" up --detach --no-build --pull never
exec /usr/bin/python "$root/router.py"
