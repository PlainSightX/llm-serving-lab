"""不能把任意失败记录冒充活动worker故障注入。"""

import importlib.util
import json
from pathlib import Path
import pytest


def test_fault_requires_exact_inflight_process_and_confirmed_recovery():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_releases.py'
    spec = importlib.util.spec_from_file_location('release_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = {'pid': 10, 'boot_id': 'boot', 'start_ticks': '100'}
    row = {
        'worker': json.dumps({'processes': [process]}),
        'state': 'failed',
        'cleanup': 'confirmed',
    }
    module.verify_fault({'signal': 'SIGKILL', 'process': process}, row)
    for kill in (
        {},
        {'signal': 'SIGTERM', 'process': process},
        {'signal': 'SIGKILL', 'process': {**process, 'start_ticks': '200'}},
    ):
        with pytest.raises(ValueError):
            module.verify_fault(kill, row)
    with pytest.raises(ValueError):
        module.verify_fault(
            {'signal': 'SIGKILL', 'process': process}, {**row, 'cleanup': 'unconfirmed'}
        )
