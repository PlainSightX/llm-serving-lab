"""候选评价必须消费已完成的固定训练，不能只和任意worker自洽。"""

import copy
import pytest


def metadata():
    from serving_lab.workflow_training import N1_RECIPE, C0_DIGEST
    from serving_lab.task_protocol import digest

    manifest = {
        'count': 1794,
        'recipe': N1_RECIPE,
        'data_sha256': 'data',
        'dev_samples': 0,
        'mix': {'dev_rows': 0, 'future_rows': 0, 'phase': 1},
    }
    initial = {
        'mode': 'workflow_stage1',
        'recipe': N1_RECIPE,
        'optimizer_steps': 675,
        'dataset_manifest_sha256': 'manifest',
        'data': manifest,
        'base_artifact_digest': C0_DIGEST,
        'base_binding_sha256': 'binding',
    }
    summary = {
        'optimizer_steps': 675,
        'epochs': 3,
        'samples': 1794,
        'data_sha256': 'data',
        'base_artifact_digest': C0_DIGEST,
        'base_binding_sha256': 'binding',
        'merged_identity': {'files': {'model.safetensors': 'candidate'}},
        'adapter_reload_max_difference': 0.0,
        'merged_reload_max_difference': 0.0,
    }
    return initial, summary, manifest, digest(summary['merged_identity']['files'])


def test_incomplete_or_unrelated_training_cannot_supply_a_candidate():
    from serving_lab.training_evidence import check_training_binding

    initial, summary, manifest, artifact = metadata()
    check_training_binding(initial, summary, manifest, 'manifest', artifact)
    for key, value in [
        ('optimizer_steps', 674),
        ('base_artifact_digest', 'other'),
        ('data_sha256', 'other'),
    ]:
        bad = copy.deepcopy(summary)
        bad[key] = value
        with pytest.raises(ValueError):
            check_training_binding(initial, bad, manifest, 'manifest', artifact)
    with pytest.raises(ValueError):
        check_training_binding(initial, summary, manifest, 'manifest', 'other artifact')
    with pytest.raises(ValueError):
        check_training_binding(initial, summary, manifest, 'different data manifest', artifact)


def test_training_log_must_contain_the_fixed_steps_and_sample_order():
    from serving_lab.training_evidence import check_training_log

    recipe = {'epochs': 1, 'accumulation': 2, 'seed': 1}
    encoded = [{'supervised_tokens': 1}, {'supervised_tokens': 2}, {'supervised_tokens': 3}]
    # Random(1)打乱后的索引为1,2,0。
    rows = [
        {
            'update': 1,
            'epoch': 1,
            'samples': [1, 2],
            'supervised_tokens_seen': 5,
            'mean_loss': 0.2,
            'elapsed_seconds': 1,
        },
        {
            'update': 2,
            'epoch': 1,
            'samples': [0],
            'supervised_tokens_seen': 6,
            'mean_loss': 0.1,
            'elapsed_seconds': 2,
        },
    ]
    assert check_training_log(rows, encoded, recipe) == 6
    with pytest.raises(ValueError):
        check_training_log(rows[:1], encoded, recipe)
    rows[1]['samples'] = [2]
    with pytest.raises(ValueError):
        check_training_log(rows, encoded, recipe)
