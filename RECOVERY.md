# Elpis automatic terminal-failure recovery — 30 October 2026

Installed as `llmrig-elpis-recovery.service` in isa's user systemd manager.
Production script: `/home/isa/Projects/llmrig/recipes/elpis-dual-262k/recovery.py`.
The router and healthy engines were not restarted to install it.

## Safety contract

- Poll every 10 seconds. Require three consecutive observations of the authenticated backend's exact sticky `503 generation_failed / Native generator requires restart` response.
- Router must report the target unhealthy and idle. Recheck under the shared media lock. Busy or healthy engines are never recovery targets.
- Acquire the existing GPU-switch lock without waiting; do not collide with vision, MinerU, OCR or ASR, including while recovering GPU0.
- Require the main service active and target container running, not paused or restarting. Maintenance stops and transient HTTP failures do not authorize a restart.
- Restart only the failed container. Hold media lock through readiness (bounded 180 seconds). Never replay an inference request.
- At most three attempts per lane per hour, separated by at least five minutes. Persist attempts before action at `~/.local/state/llmrig-elpis-recovery.json` so monitor restarts cannot reset the budget.
- Docker's existing policy handles exited containers. Unresponsive but non-terminal engines are deliberately not force-restarted: this avoids killing long in-flight work.

The backend source was inspected: `engine.failure` is assigned on native-worker exceptions and stays set; `/health` then emits the exact failure above, and later generation refuses to run until restart. This is the basis for safely handling GPU0 without altering the router's in-process locks.

## Operations

`systemctl --user status llmrig-elpis-recovery`

`journalctl --user -u llmrig-elpis-recovery`

Rollback only this addition: `systemctl --user disable --now llmrig-elpis-recovery`.
This does not stop inference engines or the router.

## Historical verification

The following records describe the October 30 host-side validation. The referenced `test_recovery.py` and raw JSON outputs are not included in this public snapshot; the result is historical, not a test rerun for publication. The included live validation scripts are `recovery-validate.py` and `recovery-validate-answer.py`; they send real requests and are not offline unit tests.

`python3 -m pytest -q test_recovery.py`: 11 passed, including actual HTTP 503-to-200 simulation with the Docker restart action mocked, active work exclusion, media lock, final recheck, maintenance exclusions, and persistent retry budget.

No OOM or deliberate failure was injected into production. Production checks use normal API requests and synthetic red-square images. Main run: `validate.py`; answer/cold-overlap run: `validate_answer.py`. `validate_cold.py` is an unused fallback variant.

Telemetry comes from `/props`, sampled every two seconds; GPU readings there can be cached for five seconds, so measured maxima can miss short peaks. Tests cover approximately 20k input tokens and a small synthetic image, not the entire 262k context or real multipage documents.

## Live results (Luxembourg time, 19:53–19:56)

- Initial short text: HTTP 200, GPU0, 2.023 s, final answer OK.
- Cold text: 20,058 prompt tokens, 13.373 s, GPU0, HTTP 200, final answer OK.
- Cached text during vision: 20,058 prompt tokens / 19,968 cached, 0.586 s, GPU0, HTTP 200.
- Cold text during final vision: 20,058 prompt tokens / zero cached, 13.644 s, GPU0, HTTP 200, final answer OK.
- First two vision requests used 64 output tokens and exhausted their budget on reasoning (HTTP 200, no final content). Corrective 256-token test returned Red, finish_reason=stop, in 10.505 s; this is a tiny synthetic square, not comparable to the report's document-page benchmarks.
- Final text after transition: HTTP 200, 0.361 s, OK. Both lanes healthy and idle at 19:56.
- Across 60 sampled snapshots GPU0 stayed healthy, but its observed free memory minimum was **427 MiB**, peak used **23,743 MiB**. Do not interpret these passing tests as full-context stability certification.
- Host available memory minimum: 17,355.7 MiB. Swap peak: 3,128.7 MiB. During the final test, about 1.95 GiB of process swap belonged to the GPU0 Elpis Python process; Quickshell processes together about 452 MiB; Ollama about 150 MiB. These are snapshots, not proof of a leak.
- Final snapshot: GPU0 1,128 MiB free; GPU1 2,264 MiB free; host 19,155 MiB available; swap 3,022.5 MiB. Short vmstat window showed zero swap-out, with some swap-in.
- GPU0 container start timestamp remained 17:32:58Z. Only normal media routing stopped/restarted lane1. The recovery monitor issued no container restarts during these healthy tests; transient ConnectionResetError during lane1 startup was deferred safely.

Recommended next work: investigate dynamic GPU0 allocations and desktop VRAM use, establish a measured memory reserve, then retest long-context workloads before altering context/cache budgets. Automatic recovery limits downtime after terminal failure; it does not prevent OOM.
