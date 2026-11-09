"""基础任务没有workflow字段，复用执行器时仍须保留用途和清理边界。"""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import pytest


@pytest.mark.asyncio
async def test_basic_episode_accepts_basic_schema_and_closes_owned_bridge(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path('scripts').resolve()))
    module = runpy.run_path('scripts/n5_qualify.py')
    episode = module['episode']
    scope = episode.__globals__
    view = {
        'feedback': 'Your task is to: put an apple in countertop.',
        'admissible_commands': ['look'],
    }

    class Bridge:
        closed = False

        def read(self):
            return {'public': view, 'terminal': False}

        async def step(self, action):
            return {'public': view, 'terminal': True, 'result': {'success': False, 'steps': 1}}

        def close(self):
            self.closed = True

    bridge = Bridge()
    observed = {}

    async def open_bridge(*args, **kwargs):
        args[2].mkdir(parents=True, exist_ok=False)
        return bridge

    async def generate(*args, **kwargs):
        observed.update(metadata=args[5], choices=kwargs['choices'])
        return {'content': 'look', 'finish_reason': 'stop'}

    monkeypatch.setitem(scope, 'open_owned_bridge', open_bridge)
    monkeypatch.setitem(scope, 'generate_action', generate)
    plan = {
        'qualification': {
            'max_steps': 100,
            'stagnation_limit': 4,
            'input_tokens': 3500,
            'generation_cap': 8000,
        }
    }
    (tmp_path / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: [1, 2, 3])
    engine = SimpleNamespace(
        identity={'id': 'unit-only'}, profile=SimpleNamespace(digest='unit-only')
    )
    row = await episode(
        engine,
        None,
        tokenizer,
        tmp_path,
        {'id': 'basic-fixture', 'condition': 'basic'},
        {'task_id': 'basic-fixture'},
        tmp_path,
        None,
        tmp_path / 'episode',
        plan,
        basic=True,
    )
    assert row['workflow'] is None and row['purpose'] == 'n5_basic_actor_readiness'
    assert observed['metadata']['purpose'] == 'n5_basic_actor_readiness' and bridge.closed
    assert not row['eligible_for_training'] and not row['eligible_for_primary_history']
