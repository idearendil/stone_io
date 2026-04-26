import json
import os
import subprocess
import time
import urllib.error
import urllib.request

_SERVER_JS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'headless', 'HeadlessServer.js')
)
_TIMEOUT = 5.0


class HeadlessBridge:
    """Launches and communicates with the Node.js HeadlessServer subprocess."""

    def __init__(self, port: int = 7777, num_agents: int = 1, num_bots: int = 0):
        self.port = port
        self._base = f'http://127.0.0.1:{port}'
        self._proc: subprocess.Popen | None = None
        self._launch(num_agents, num_bots)

    def _launch(self, num_agents: int, num_bots: int) -> None:
        env = os.environ.copy()
        env['PORT']       = str(self.port)
        env['NUM_AGENTS'] = str(num_agents)
        env['NUM_BOTS']   = str(num_bots)
        self._proc = subprocess.Popen(
            ['node', _SERVER_JS],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        for attempt in range(5):
            try:
                req = urllib.request.Request(f'{self._base}/ping')
                with urllib.request.urlopen(req, timeout=_TIMEOUT):
                    return
            except urllib.error.HTTPError:
                return  # got a real HTTP response — server is up
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f'HeadlessServer did not respond on port {self.port} after 5 attempts'
        )

    def post(self, path: str, data: dict | None = None) -> dict:
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            f'{self._base}{path}',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())

    def get(self, path: str) -> dict:
        req = urllib.request.Request(f'{self._base}{path}')
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
