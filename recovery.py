"""Recover sticky native failures without restarting the router or healthy engines.

Fail closed: timeouts, unknown errors, stopped containers and unavailable router
never authorize a restart. Native generation_failed is sticky until process exit.
"""
import fcntl
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG = logging.getLogger('elpis-recovery')
ROOT = Path('/home/isa/Projects/llmrig/recipes/elpis-dual-262k')
STATE = Path.home() / '.local/state/llmrig-elpis-recovery.json'


def get(port, path, auth=None):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                 headers={'Authorization': auth} if auth else {})
    try:
        response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def fatal(status, body):
    error = body.get('error', {})
    return (status == 503 and isinstance(error, dict)
            and error.get('code') == 'generation_failed'
            and error.get('message') == 'Native generator requires restart')


def eligible(slot):
    return not slot['healthy'] and not slot['busy'] and not slot['gpu_lease_held']


class Recovery:
    def __init__(self, config, state=STATE):
        self.cfg = config
        self.auth = 'Bearer ' + Path(config['api_key_file']).read_text().strip()
        self.state = state
        self.failures = [0, 0]
        self.attempts = json.loads(state.read_text()) if state.exists() else {'0': [], '1': []}

    def active(self):
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                               'llmrig-elpis.service']).returncode == 0

    def slots(self):
        status, body = get(self.cfg.get('port', 11446), '/props')
        if status != 200:
            raise RuntimeError('Router telemetry unavailable')
        return body['slots']

    def check(self, i):
        return fatal(*get(self.cfg['ports'][i], '/health', self.auth))

    def tick(self):
        if not self.active():
            self.failures = [0, 0]
            return
        slots = self.slots()
        for i in range(2):
            if not eligible(slots[i]) or not self.check(i):
                self.failures[i] = 0
                continue
            self.failures[i] += 1
            if self.failures[i] < 3:
                continue
            now = time.time()
            history = self.attempts[str(i)] = [t for t in self.attempts[str(i)] if now-t < 3600]
            if len(history) >= 3 or (history and now-history[-1] < 300):
                continue
            # Shared with router vision, PDF, OCR and ASR. Never wait on their jobs.
            with open(self.cfg['media_lock'], 'a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Our own lock sets gpu_lease_held for lane1; check busy/health
                # directly while holding it. Router cannot admit lane1 work now.
                fresh = self.slots()[i]
                if fresh['healthy'] or fresh['busy'] or not self.active() or not self.check(i):
                    continue
                name = f'llmrig-elpis-lane{i}'
                inspected = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                                           capture_output=True, text=True, check=True, timeout=10)
                container = json.loads(inspected.stdout)
                if not container['Running'] or container.get('Paused') or container.get('Restarting'):
                    continue
                # Persist BEFORE action: monitor restarts cannot reset rate limits.
                history.append(now)
                self.state.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.state.with_suffix('.new')
                temporary.write_text(json.dumps(self.attempts))
                temporary.replace(self.state)
                LOG.warning('Restarting terminally failed idle lane=%s attempt=%s/3 per hour', i, len(history))
                subprocess.run(['docker', 'restart', '--time', '45', name],
                               check=True, capture_output=True, timeout=90)
                self.failures[i] = 0
                # Keep media excluded until restored, or bounded timeout.
                deadline = time.monotonic()+180
                while time.monotonic() < deadline:
                    try:
                        status, _ = get(self.cfg['ports'][i], '/health', self.auth)
                        if status == 200:
                            LOG.warning('Recovered lane=%s', i)
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(5)
                else:
                    LOG.error('lane=%s did not recover within 180 seconds', i)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    monitor = Recovery(json.loads((ROOT / 'router.json').read_text()))
    while True:
        try:
            monitor.tick()
        except Exception as error:
            monitor.failures = [0, 0]
            LOG.warning('Recovery deferred: %s', type(error).__name__)
        time.sleep(10)
