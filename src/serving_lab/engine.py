"""受控引擎实例的身份与释放证据；不负责实验评价或模型训练。"""

from __future__ import annotations

import json
import asyncio
import contextlib
import socket
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .lab import digest, file_hash


def check_port_available(host, port):
    """按Linux服务器的重绑定语义检查，不把TIME_WAIT当成存活监听者。"""
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))


def process_identity(pid, *, proc_root=Path('/proc')):
    """boot_id与start_ticks共同区分PID复用；stat中的进程名允许空格。"""
    try:
        text = (Path(proc_root) / str(pid) / 'stat').read_text()
        fields = text[text.rindex(')') + 2 :].split()
        if fields[0] == 'Z':
            return None
        return {
            'pid': pid,
            'start_ticks': fields[19],
            'boot_id': (Path(proc_root) / 'sys/kernel/random/boot_id').read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class EngineBinding:
    version_id: str
    endpoint: str
    served_model: str
    artifact_digest: str
    worker_instance_id: str
    startup_generation: str
    process: dict
    model_path: str
    events_path: str
    manifest_path: Path
    manifest_sha256: str
    launch_sha256: str

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        endpoint = urlsplit(data['endpoint'])
        if (
            endpoint.scheme != 'http'
            or endpoint.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or not endpoint.port
            or endpoint.path not in {'', '/'}
            or endpoint.username
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError('R1只允许固定loopback引擎')
        launch_path = path.parent / 'launch.json'
        launch = json.loads(launch_path.read_text(encoding='utf-8'))
        config = launch['config']
        command = launch['command']
        try:
            consistent = (
                launch['worker_instance_id'] == data['worker_instance_id']
                and launch['startup_generation'] == data['startup_generation']
                and data['version_id'] == data['served_model'] == config['version_id']
                and data['endpoint'].rstrip('/') == f'http://{config["host"]}:{config["port"]}'
                and Path(data['model_path']).resolve()
                == Path(launch['base_identity']['path']).resolve()
                and data['artifact_digest'] == digest(launch['base_identity']['files'])
                and command[command.index('--model') + 1] == data['model_path']
                and command[command.index('--served-model-name') + 1] == data['served_model']
                and command[command.index('--host') + 1] == config['host']
                and command[command.index('--port') + 1] == str(config['port'])
            )
        except (KeyError, IndexError, ValueError):
            consistent = False
        if not consistent:
            raise ValueError('binding与实际启动清单不一致')
        identity_path = path.parent / 'runtime-identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8'))[
            'binding_sha256'
        ] != file_hash(path):
            raise ValueError('binding相对首次控制者身份发生漂移')
        fields = {
            k: data[k]
            for k in cls.__dataclass_fields__
            if k not in {'manifest_path', 'manifest_sha256', 'launch_sha256'}
        }
        return cls(
            **fields,
            manifest_path=path,
            manifest_sha256=file_hash(path),
            launch_sha256=file_hash(launch_path),
        )

    def manifest_unchanged(self):
        try:
            return (
                file_hash(self.manifest_path) == self.manifest_sha256
                and file_hash(self.manifest_path.parent / 'launch.json') == self.launch_sha256
            )
        except OSError:
            return False

    def alive(self):
        return self.manifest_unchanged() and process_identity(self.process['pid']) == self.process


class ReleaseJournal:
    """增量读当前实例日志；不让历史代际的同名请求满足清理确认。"""

    def __init__(self, path, instance, generation, *, max_entries=4096):
        self.path = Path(path)
        self.instance = instance
        self.generation = generation
        self.max_entries = max_entries
        self.offset = 0
        self.pending = b''
        self.released = OrderedDict()
        self.installed_pids = set()
        self.scheduler_pids = set()

    def refresh(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise RuntimeError('释放证据被截断')
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while data := stream.read(65536):
                self.offset = stream.tell()
                self._consume(data)

    def _consume(self, data):
        lines = (self.pending + data).split(b'\n')
        self.pending = lines.pop()
        if len(self.pending) > 1024 * 1024:
            raise RuntimeError('释放事件异常超长')
        for line in lines:
            if not line:
                continue
            event = json.loads(line)
            if (
                event.get('worker_instance_id') != self.instance
                or event.get('startup_generation') != self.generation
            ):
                continue
            if event['event'] == 'probe_installed':
                self.installed_pids.add(event['pid'])
            if (
                event['event'] == 'request_released'
                and event['request_absent']
                and not any(event['after_blocks'])
            ):
                self.released[event['request_id']] = event
                self.released.move_to_end(event['request_id'])
                if len(self.released) > self.max_entries:
                    self.released.popitem(last=False)
                if event.get('pid') is not None:
                    self.scheduler_pids.add(event['pid'])

    def get(self, request_id):
        self.refresh()
        prefix = 'chatcmpl-' + request_id
        return self.released.get(prefix) or self.released.get(prefix + '-0')

    def find_historical(self, request_id):
        """仅恢复时扫描旧日志；在线清理查询使用有界近期索引。"""
        if not self.path.exists():
            return None
        wanted = {'chatcmpl-' + request_id, 'chatcmpl-' + request_id + '-0'}
        with self.path.open('rb') as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):
                    break
                event = json.loads(raw)
                if (
                    event.get('event') == 'request_released'
                    and event.get('request_id') in wanted
                    and event.get('worker_instance_id') == self.instance
                    and event.get('startup_generation') == self.generation
                    and event.get('request_absent')
                    and not any(event['after_blocks'])
                ):
                    return event
        return None


class ReadinessCache:
    """慢就绪检查（清单哈希、网络健康探测）的结果缓存，可由后台主动探测保持新鲜。

    只缓存“通过”的结果，有效期从探测开始时刻算起；失败、超时或身份变化立即作废。
    同一时刻只有一次在途探测，所有等待者共享它的结果（无论成败），失败的探测不会被排队重试。
    启动后台探测后，请求路径通常只读结果，与请求频率无关；未启动或结果过期时等待（或发起）
    一次探测。有效期内的引擎挂起由请求期限兜底（504）；进程或代际变化由调用方每次单独核对。
    """

    def __init__(self, ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self.passed_at = None
        self.inflight = None
        self.probes = 0
        self.monitor = None

    def fresh(self):
        return self.passed_at is not None and time.monotonic() - self.passed_at <= self.ttl_seconds

    def invalidate(self):
        self.passed_at = None

    async def _run(self, probe):
        self.probes += 1
        started = time.monotonic()
        try:
            passed = bool(await probe())
        except Exception:
            passed = False
        self.passed_at = started if passed else None
        return passed

    async def check(self, probe, *, force=False):
        if not force and self.fresh():
            return True
        if self.inflight is None or self.inflight.done():
            task = asyncio.ensure_future(self._run(probe))
            self.inflight = task
            task.add_done_callback(
                lambda done: setattr(self, 'inflight', None) if self.inflight is done else None
            )
        # shield：某个等待者被取消（如客户端断开）时，不取消其他人共享的探测。
        return await asyncio.shield(self.inflight)

    def start(self, probe, interval_seconds):
        """后台按固定间隔主动探测；间隔须小于有效期，正常运行时结果始终新鲜。"""
        if interval_seconds >= self.ttl_seconds:
            raise ValueError('探测间隔必须小于结果有效期')
        if self.monitor is not None:
            return

        async def loop():
            while True:
                await self.check(probe, force=True)
                await asyncio.sleep(interval_seconds)

        self.monitor = asyncio.create_task(loop())

    def monitoring(self):
        return self.monitor is not None and not self.monitor.done()

    async def stop(self):
        if self.monitor is not None:
            self.monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor
            self.monitor = None
        if self.inflight is not None and not self.inflight.done():
            self.inflight.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.inflight


class HttpEngine:
    """固定本机实例适配；线程池仅做有界token计数，引擎释放以插件事件确认。"""

    # 清单哈希与HTTP /health 由后台每 0.2 秒探测一次，通过结果最多复用 0.5 秒；
    # 进程身份每次请求都核对。
    HEALTH_TTL_SECONDS = 0.5
    HEALTH_INTERVAL_SECONDS = 0.2

    def __init__(self, binding, *, profile=False, profile_policy=None):
        import httpx
        from transformers import AutoTokenizer

        self.binding = binding
        self.version_id = binding.version_id
        self.journal = ReleaseJournal(
            binding.events_path, binding.worker_instance_id, binding.startup_generation
        )
        self.journal.refresh()
        if not self.journal.installed_pids:
            raise RuntimeError('未观察到引擎释放探针')
        identity_path = binding.manifest_path.parent / 'runtime-identity.json'
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            if (
                identity['worker_instance_id'] != binding.worker_instance_id
                or identity['startup_generation'] != binding.startup_generation
                or identity['binding_sha256'] != binding.manifest_sha256
            ):
                raise RuntimeError('实例身份文件不符')
        else:
            # 模型架构探测子进程也会安装插件；只有真实释放过请求的进程证明Scheduler实际运行。
            if not self.journal.scheduler_pids:
                raise RuntimeError('先完成直连请求及真实Scheduler释放资格')
            processes = [binding.process] + [
                process_identity(pid)
                for pid in sorted(self.journal.scheduler_pids)
                if pid != binding.process['pid']
            ]
            if not all(processes):
                raise RuntimeError('实际worker进程已变化')
            identity = {
                'worker_instance_id': binding.worker_instance_id,
                'startup_generation': binding.startup_generation,
                'binding_sha256': binding.manifest_sha256,
                'processes': processes,
            }
            from .execution import write_once

            write_once(identity_path, identity)
        self.identity = identity
        self.tokenizer = AutoTokenizer.from_pretrained(
            binding.model_path, local_files_only=True, trust_remote_code=False
        )
        from .profile import make_profile

        if profile_policy is not None and not profile:
            raise ValueError('研究策略须显式profile')
        self.profile = make_profile(binding, self.tokenizer, profile_policy) if profile else None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='serving-tokenizer')
        self.client = httpx.AsyncClient(
            base_url=binding.endpoint,
            timeout=None,
            trust_env=False,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
        self.readiness = ReadinessCache(self.HEALTH_TTL_SECONDS)

    def _identity_ok(self):
        """worker 主进程与 Scheduler 进程身份（读 /proc，微秒级）；同端口被替代实例接管时立即失败。"""
        if process_identity(self.binding.process['pid']) != self.binding.process or not all(
            process_identity(p['pid']) == p for p in self.identity['processes']
        ):
            self.readiness.invalidate()
            return False
        return True

    async def ready(self):
        if not self._identity_ok():
            return False
        return await self.readiness.check(self._slow_ready)

    def ready_nowait(self):
        """后台探测运行时不等待：返回进程身份 + 最近一次探测结果；未运行时返回 None。"""
        if not self.readiness.monitoring():
            return None
        return self._identity_ok() and self.readiness.fresh()

    async def _slow_ready(self):
        """较慢的检查：启动清单未被改动，且引擎 HTTP /health 正常。"""
        # 清单哈希是同步文件读取（清单在 Windows 挂载盘上时约数毫秒），放到线程执行，避免阻塞事件循环。
        if not await asyncio.to_thread(self.binding.manifest_unchanged):
            return False
        return (await self.client.get('/health', timeout=1)).status_code == 200

    def start_monitoring(self):
        """在服务事件循环内启动后台健康探测（由网关启动时调用）。"""
        self.readiness.start(self._slow_ready, self.HEALTH_INTERVAL_SECONDS)

    async def count_tokens(self, messages):
        return len(await self.token_ids(messages))

    async def token_ids(self, messages):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            ),
        )

    async def engine_token_ids(self, messages):
        result = await self.client.post(
            '/tokenize',
            json={
                'model': self.binding.served_model,
                'messages': messages,
                'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False},
            },
            timeout=5,
        )
        result.raise_for_status()
        return result.json()['tokens']

    def wire_payload(self, body):
        if self.profile:
            return self.profile.payload(body)
        payload = {
            **body,
            'model': self.binding.served_model,
            'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        if body.get('stream'):
            payload['stream_options'] = {'include_usage': True}
        return payload

    async def open(self, attempt, body):
        payload = self.wire_payload(body)
        req = self.client.build_request(
            'POST', '/v1/chat/completions', json=payload, headers={'X-Request-Id': attempt}
        )
        return await self.client.send(req, stream=True)

    async def wait_release(self, attempt, timeout):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            event = self.journal.get(attempt)
            if event:
                return event
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.02)

    async def reconcile(self, attempt, old_identity, timeout):
        if (old_identity.get('worker_instance_id'), old_identity.get('startup_generation')) == (
            self.identity['worker_instance_id'],
            self.identity['startup_generation'],
        ):
            return self.journal.find_historical(attempt) or await self.wait_release(
                attempt, timeout
            )
        processes = old_identity.get('processes', [])
        if processes and all(process_identity(p['pid']) != p for p in processes):
            return {'status': 'WORKER_GENERATION_TERMINATED', 'processes': processes}
        return None

    async def close(self):
        await self.readiness.stop()
        await self.client.aclose()
        self.executor.shutdown(wait=False, cancel_futures=True)
