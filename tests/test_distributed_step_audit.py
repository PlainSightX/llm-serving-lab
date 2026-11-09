"""完整步序与分布式样本必须来自实际记录，摘要不能掩盖缺步或重复。"""

from copy import deepcopy
import pytest


def test_training_log_rejects_missing_duplicate_and_wrong_rank_samples():
    from serving_lab.distributed_study import validate_training_steps

    records = [{'id': str(i), 'input_ids': [1, 2, 3, 4]} for i in range(8)]
    contract = {
        'planned_steps': 2,
        'global_batch': 4,
        'world_size': 2,
        'sequence_length': 4,
        'phase': 'correctness',
        'profile_enabled': False,
    }
    rows = [
        {
            'step': s + 1,
            'rank': 1,
            'sample_ids': [str(s * 4 + 1), str(s * 4 + 3)],
            'global_sample_ids': [str(i) for i in range(s * 4, s * 4 + 4)],
            'global_valid_tokens': 12,
            'loss_multiplier': 2 / 12,
            'processed_tokens': 16,
            'loss': 1.0,
            'wall_seconds': 0.1,
            'cuda_event_seconds': 0.09,
            'peak_allocated_bytes': 100,
            'peak_reserved_bytes': 200,
            'measured': False,
            'profiled': False,
        }
        for s in range(2)
    ]
    assert validate_training_steps(rows, contract, records, rank=1, start_step=0, warmup=0) == [
        1,
        2,
    ]
    for broken in [rows[:1], [rows[0], rows[0]], list(reversed(rows))]:
        with pytest.raises(ValueError):
            validate_training_steps(broken, contract, records, rank=1, start_step=0, warmup=0)
    broken = deepcopy(rows)
    broken[1]['sample_ids'] = ['4', '6']
    with pytest.raises(ValueError):
        validate_training_steps(broken, contract, records, rank=1, start_step=0, warmup=0)


def test_resume_audit_requires_exact_remaining_steps():
    from serving_lab.distributed_study import validate_training_steps

    with pytest.raises(ValueError):
        validate_training_steps([], {'planned_steps': 12}, [], rank=0, start_step=4, warmup=4)


def test_empty_final_state_is_not_numerical_equivalence():
    import runpy

    compare = runpy.run_path('scripts/n4_compare.py')
    with pytest.raises(ValueError):
        compare['tensor_metrics']({}, {})


def test_self_consistent_shortened_run_cannot_replace_frozen_plan():
    from serving_lab.distributed_study import validate_training_contract

    plan = {
        'global_batch': 4,
        'model_artifact_digest': 'a',
        'performance': {
            'sequence_length': 512,
            'precision': 'bf16',
            'optimizer': 'adamw',
            'warmup_steps': 4,
            'measured_steps': 16,
        },
    }
    contract = {
        'schema': 1,
        'plan_sha256': 'p',
        'model_digest': 'a',
        'mode': 'ddp',
        'phase': 'performance',
        'world_size': 2,
        'precision': 'bf16',
        'global_batch': 4,
        'scaling': 'strong',
        'sequence_length': 512,
        'optimizer': 'adamw',
        'profile_enabled': False,
        'planned_steps': 20,
    }
    cfg = plan['performance']
    assert validate_training_contract(contract, cfg, plan, 'p') == cfg
    with pytest.raises(ValueError):
        validate_training_contract({**contract, 'planned_steps': 5}, cfg, plan, 'p')
    with pytest.raises(ValueError):
        validate_training_contract(contract, {**cfg, 'warmup_steps': 0}, plan, 'p')
    with pytest.raises(ValueError):
        validate_training_contract({**contract, 'global_batch': 8}, cfg, plan, 'p')
