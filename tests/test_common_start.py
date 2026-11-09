"""基础适配不能用相同分词器掩盖换基座或无前置资格结论。"""

import pytest


def test_common_basic_parent_is_exact_original4b():
    from serving_lab.common_start import require_parent, BASE_ARTIFACT, BASE_PARAMETER

    require_parent(BASE_ARTIFACT, BASE_PARAMETER)
    with pytest.raises(ValueError):
        require_parent('same-tokenizer-other-artifact', BASE_PARAMETER)
    with pytest.raises(ValueError):
        require_parent(BASE_ARTIFACT, 'other-parameters')


def test_common_basic_rejects_failure_from_a_different_protocol(tmp_path):
    import json
    from serving_lab.lab import file_hash
    from serving_lab.common_start import check_prerequisite, BASE_ARTIFACT, BASE_PARAMETER

    summary = {
        'completed': True,
        'qualified': False,
        'artifact_digest': BASE_ARTIFACT,
        'parameter_digest': BASE_PARAMETER,
    }
    initial = {'actor_role': 'Qwen3-4B-fixed-original', 'plan_sha256': 'different-contract'}
    for name, value in [('summary.json', summary), ('input.json', initial)]:
        (tmp_path / name).write_text(json.dumps(value), encoding='utf-8')
    audit = {
        'passed': True,
        'qualified': False,
        'actor_role': 'Qwen3-4B-fixed-original',
        'summary_sha256': file_hash(tmp_path / 'summary.json'),
        'input_sha256': file_hash(tmp_path / 'input.json'),
    }
    (tmp_path / 'audit.json').write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError):
        check_prerequisite(tmp_path)


def test_common_basic_needs_both_prior_roles(tmp_path):
    from serving_lab.common_start import check_prerequisites

    (tmp_path / 'qwen4').mkdir()
    (tmp_path / 'qwen4/audit.json').write_text('{}')
    with pytest.raises(ValueError):
        check_prerequisites(tmp_path)


def test_changed_raw_source_cannot_follow_an_old_repair_card():
    from serving_lab.common_start import (
        validate_plan,
        COMMON_RECIPE,
        BASE_ARTIFACT,
        BASE_PARAMETER,
        BASE_REVISION,
        DATA,
        COLLECTION,
    )

    plan = {
        'name': 'n5-common-4b-basic-repair-v1',
        'recipe': COMMON_RECIPE,
        'parent_artifact_digest': BASE_ARTIFACT,
        'parent_parameter_digest': BASE_PARAMETER,
        'parent_revision': BASE_REVISION,
        'source_data': DATA,
        'source_collection': COLLECTION,
        'recipes_allowed': 1,
        'final_checkpoint_only': True,
        'source_manifest_sha256': 'a' * 64,
        'source_collection_sha256': 'b' * 64,
    }
    validate_plan(plan, {'manifest': 'a' * 64, 'collection': 'b' * 64})
    with pytest.raises(ValueError):
        validate_plan(plan, {'manifest': 'c' * 64, 'collection': 'b' * 64})
