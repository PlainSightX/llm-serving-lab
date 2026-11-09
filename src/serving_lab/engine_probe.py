"""固定vLLM版本的只读释放观测；不改变调度选择、KV分配或生成结果。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def install_release_probe(scheduler_class, path, instance, generation):
    if getattr(scheduler_class, '_serving_lab_release_probe', False):
        return
    original = scheduler_class._free_blocks
    path = Path(path)

    def emit(event):
        event.update(
            worker_instance_id=instance,
            startup_generation=generation,
            pid=os.getpid(),
            wall_ns=time.time_ns(),
        )
        try:
            # 单次O_APPEND写入，不让观测写入失败改变引擎行为；缺事件由控制者判为未确认。
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (json.dumps(event, separators=(',', ':')) + '\n').encode())
            finally:
                os.close(fd)
        except OSError:
            pass

    def observed(self, request):
        request_id = request.request_id
        before = self.kv_cache_manager.get_block_ids(request_id)
        result = original(self, request)
        after = self.kv_cache_manager.get_block_ids(request_id)
        status = getattr(request.status, 'name', str(request.status))
        emit(
            {
                'event': 'request_released',
                'request_id': request_id,
                'status': status,
                'before_blocks': before,
                'after_blocks': after,
                'request_absent': request_id not in self.requests,
            }
        )
        return result

    scheduler_class._free_blocks = observed
    scheduler_class._serving_lab_release_probe = True
    emit({'event': 'probe_installed'})


def register():
    """仅由明确启用的vllm.general_plugins入口调用；核心环境无需导入vLLM。"""
    path = os.environ.get('SERVING_LAB_RELEASE_EVENTS')
    if not path:
        return
    import importlib.metadata

    if importlib.metadata.version('vllm') != '0.11.2':
        raise RuntimeError('释放探针仅核验vLLM 0.11.2')
    from vllm.v1.core.sched.scheduler import Scheduler

    install_release_probe(
        Scheduler, path, os.environ['SERVING_LAB_WORKER_ID'], os.environ['SERVING_LAB_GENERATION']
    )
