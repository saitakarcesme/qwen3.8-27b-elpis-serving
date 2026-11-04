"""Bounded, exclusive document sessions. No prompt/image logging or file deletion."""
from __future__ import annotations
import contextlib
import fcntl
import http.client
import json
import logging
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

LOG = logging.getLogger('media-session')

class SessionError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)

class MediaSession:
    def __init__(self, server, *, clock=time.monotonic, start_watchdog=True):
        self.server = server
        self.cfg = server.config
        self.clock = clock
        self.mu = threading.RLock()
        self.work = threading.Lock()
        self.current = None
        self.connection = None
        self.client_connection = None
        self.upstream_socket = None
        self.worker_running = False
        self.stopping = threading.Event()
        self.state_path = Path(self.cfg.get('media_session_state', str(Path(__file__).with_name('media-session-state.json'))))
        if start_watchdog:
            threading.Thread(target=self.watch, daemon=True, name='media-session-watchdog').start()

    def persist(self, active):
        # Deliberately retain the record after close; no unlink or model-file changes.
        with self.state_path.open('w') as f:
            f.write(json.dumps({'active':active,'model':self.cfg['multimodal_model'],'updated_at':time.time()})+'\n')
            f.flush()
            import os
            os.fsync(f.fileno())

    def event(self, event, **fields):
        LOG.info('media_session event=%s %s', event, json.dumps(fields, sort_keys=True))

    def view(self, token):
        with self.mu:
            s = self.current
            if not s or not secrets.compare_digest(s['id'], token):
                raise SessionError(404, 'Unknown media session')
            return {k:s[k] for k in ['id','state','idle_timeout_seconds','max_duration_seconds','reason','requests','last_error']} | {
                'model': self.cfg['multimodal_model'],
                'hard_deadline_remaining_seconds':max(0, round(s['deadline']-self.clock(), 3)),
                'lane1_ready':s['state']=='closed',
            }

    def occupied(self):
        with self.mu:
            return self.current is not None and self.current['state'] != 'closed'

    @staticmethod
    def limits(body):
        if set(body)-{'idle_timeout_seconds','max_duration_seconds'}:
            raise SessionError(400, 'Only idle_timeout_seconds and max_duration_seconds are accepted')
        idle=body.get('idle_timeout_seconds',120); cap=body.get('max_duration_seconds',1800)
        if type(idle) is not int or not 5 <= idle <= 120:
            raise SessionError(400,'idle_timeout_seconds must be an integer from 5 to 120')
        if type(cap) is not int or not 10 <= cap <= 1800:
            raise SessionError(400,'max_duration_seconds must be an integer from 10 to 1800')
        return idle,cap

    def docker(self, action, container):
        args=['docker',action]
        if action=='stop': args += ['--time','10']
        subprocess.run(args+[container],check=True,capture_output=True,timeout=30)

    def ollama(self, path, body=None, timeout=15, cancellable=False):
        c=http.client.HTTPConnection('127.0.0.1',self.cfg.get('vision_port',11436),timeout=timeout)
        try:
            c.connect()
            if cancellable: self.attach(c)
            c.request('POST' if body is not None else 'GET',path,
                      json.dumps(body) if body is not None else None,
                      {'Content-Type':'application/json'} if body is not None else {})
            r=c.getresponse();raw=r.read(1024*1024)
            if r.status!=200:raise RuntimeError('Ollama status '+str(r.status))
            return json.loads(raw)
        finally:
            c.close()
            with self.mu:
                if self.connection is c:self.connection=None

    def attach(self,c,client=None):
        with self.mu:
            if not self.current or self.current['state']=='closing':
                raise SessionError(410,'Media session is closing')
            self.connection=c
            self.upstream_socket=c.sock
            self.client_connection=client

    def open(self,body):
        idle,cap=self.limits(body)
        with self.mu:
            if self.occupied():raise SessionError(409,'A media session is already active or restoring lane 1')
            if not self.work.acquire(blocking=False):raise SessionError(409,'Media transition in progress')
            try:
                index=self.server.pool.reserve(self.cfg.get('vision_lane',1),timeout=0)
            except Exception:
                self.work.release()
                raise SessionError(409,'GPU1 is busy; retry after the current job')
            now=self.clock()
            self.current={'id':secrets.token_urlsafe(32),'state':'opening','idle_timeout_seconds':idle,
                          'max_duration_seconds':cap,'deadline':now+cap,'last_activity':now,
                          'busy':True,'index':index,'reason':None,'requests':0,'last_error':None}
            token=self.current['id']
        try:
            # Persist before stopping lane 1, so startup recovery can detect interruption.
            self.persist(True)
            self.server.pool.set_health(index,False)
            t=self.clock();self.docker('stop',self.cfg['vision_container'])
            self.event('lane_stopped',elapsed=self.clock()-t)
            self.ollama('/api/generate',{'model':self.cfg['multimodal_model'],'keep_alive':-1,
                        'options':{'num_ctx':65536}},timeout=max(1,min(120,cap)),cancellable=True)
            with self.mu:
                if self.current['state']=='closing' or self.clock()>=self.current['deadline']:
                    raise SessionError(410,'Media session expired while opening')
                self.current['state']='open';self.current['last_activity']=self.clock()
                self.current['busy']=False
            self.event('opened',elapsed=self.clock()-t,idle_seconds=idle,cap_seconds=cap)
            return self.view(token)
        except Exception:
            self.close(token,'open_failed')
            raise
        finally:self.work.release()

    @contextlib.contextmanager
    def request(self,token,payload):
        with self.mu:
            self.view(token)
            s=self.current
            if s['state']!='open' or self.clock()>=s['deadline'] or (not s['busy'] and self.clock()-s['last_activity']>=s['idle_timeout_seconds']):
                raise SessionError(410,'Media session has expired or is closing')
            if s['busy'] or not self.work.acquire(blocking=False):raise SessionError(409,'Only one vision call may run per session')
            s['busy']=True
        start=self.clock()
        try:
            if payload.get('model')!=self.cfg['multimodal_model']:
                raise SessionError(400,'Session model cannot change')
            # v0.33.3 OpenAI compatibility ignores native options/keep_alive.
            # The model's Modelfile fixes 64K context; daemon idle retention (300s)
            # exceeds our maximum idle timeout (120s). Active requests stay resident.
            if 'options' in payload or 'num_ctx' in payload:
                raise SessionError(400,'OpenAI session uses model defaults; native options/num_ctx are not accepted')
            payload.pop('keep_alive',None)
            yield max(0.1,s['deadline']-self.clock())
        finally:
            with self.mu:
                self.connection=None;self.upstream_socket=None;self.client_connection=None;s['busy']=False;s['last_activity']=self.clock();s['requests']+=1
            self.work.release()
            self.event('request_finished',elapsed=self.clock()-start,number=s['requests'])

    def close(self,token,reason='explicit_close'):
        with self.mu:
            self.view(token)
            if self.current['state']=='closed':return self.view(token)
            self.current['state']='closing';self.current['reason']=self.current['reason'] or reason
            if self.upstream_socket:
                try:self.upstream_socket.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            if self.client_connection:
                try:self.client_connection.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            if not self.worker_running:
                self.worker_running=True
                threading.Thread(target=self.restore,daemon=True,name='media-session-restore').start()
            return self.view(token)

    def ready(self):
        index=self.current['index']
        c=http.client.HTTPConnection('127.0.0.1',self.cfg['ports'][index],timeout=3)
        try:
            c.request('GET','/health',headers={'Authorization':self.server.auth})
            r=c.getresponse();r.read();return r.status==200
        except (OSError,http.client.HTTPException):return False
        finally:c.close()

    def restore(self):
        token=self.current['id']
        try:
            with self.work:
                t=self.clock();fallback=False
                try:
                    self.ollama('/api/generate',{'model':self.cfg['multimodal_model'],'keep_alive':0},timeout=10)
                    end=self.clock()+10
                    while self.ollama('/api/ps').get('models'):
                        if self.clock()>=end:raise TimeoutError('Ollama did not unload')
                        time.sleep(.25)
                except Exception:
                    # Terminate only the Ollama runtime if normal cancellation/unload hangs.
                    # Container and all mounted model files remain intact.
                    self.docker('stop','homelab-ollama');fallback=True
                self.event('unloaded',elapsed=self.clock()-t,fallback_container_stop=fallback)
                self.docker('start',self.cfg['vision_container'])
                if fallback:self.docker('start','homelab-ollama')
                self.event('lane_started',elapsed=self.clock()-t)
                end=self.clock()+180
                while not self.ready():
                    if self.clock()>=end:raise TimeoutError('Lane 1 readiness timeout; restoration will retry')
                    time.sleep(.5)
                # Keep the GPU lease until the text engine is confirmed healthy.
                self.persist(False)
                with self.mu:
                    s=self.current
                    self.server.pool.set_health(s['index'],True)
                    self.server.pool.release(s['index'])
                    s['state']='closed';s['last_error']=None;self.worker_running=False
                self.event('closed',elapsed=self.clock()-t,reason=s['reason'])
        except Exception as exc:
            with self.mu:self.current['last_error']=type(exc).__name__
            LOG.exception('Media restoration failed; retained lease, will retry')
        finally:
            with self.mu:
                if self.current['id']==token:self.worker_running=False

    def tick(self):
        with self.mu:
            s=self.current
            if not s or s['state']=='closed':return
            now=self.clock()
            if s['state']=='closing':reason=s['reason']
            elif now>=s['deadline']:reason='hard_timeout'
            elif s['state']=='open' and not s['busy'] and now-s['last_activity']>=s['idle_timeout_seconds']:reason='idle_timeout'
            else:return
            token=s['id']
        self.close(token,reason)

    def watch(self):
        while not self.stopping.wait(1):
            try:self.tick()
            except Exception:LOG.exception('Media session watchdog error')


def recover_before_start(config):
    """After router/service interruption, free orphaned media before Elpis starts."""
    path=Path(config.get('media_session_state',str(Path(__file__).with_name('media-session-state.json'))))
    if not path.exists():return
    try:active=json.loads(path.read_text()).get('active')
    except (ValueError,OSError):active=True
    if not active:return
    with open(config['media_lock'],'a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        # This runs only for a recorded unfinished session, before compose starts text lanes.
        subprocess.run(['docker','stop','--time','10','homelab-ollama'],check=True,capture_output=True,timeout=30)
        path.write_text(json.dumps({'active':False,'recovered_at':time.time()})+'\n')
        LOG.warning('Recovered orphaned media session before starting text engines')

if __name__=='__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    recover_before_start(json.loads(Path(sys.argv[1]).read_text()))
