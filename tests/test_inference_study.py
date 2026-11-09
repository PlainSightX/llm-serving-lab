"""质量面板保留错误与截断，答案不进入待评价消息。"""

import pytest


def test_numeric_answer_requires_the_final_marker_and_exact_value():
    from serving_lab.inference_study import numeric_answer, quality_pass

    assert numeric_answer('steps\n#### 1,200.00') == numeric_answer('#### 1200')
    assert numeric_answer('The answer is 1200') is None
    assert numeric_answer('#### 12\nmore text') is None
    assert quality_pass('#### 12', '#### 12', 'stop')
    assert not quality_pass('#### 12', '#### 12', 'length')
    assert not quality_pass('#### 13', '#### 12', 'stop')


def test_benchmark_payload_contains_no_evaluation_answer():
    from serving_lab.inference_study import quality_messages

    row = {'question': 'How many apples?', 'answer': 'private label #### 999999'}
    messages = quality_messages(row)
    assert '999999' not in str(messages) and 'private label' not in str(messages)
    assert messages[-1]['content'] == 'How many apples?'


@pytest.mark.asyncio
async def test_source_snapshot_does_not_overwrite_the_panel_filename(tmp_path, monkeypatch):
    import importlib.util, json, sys
    from pathlib import Path
    from types import SimpleNamespace
    from serving_lab.lab import file_hash

    path = Path(__file__).resolve().parents[1] / 'scripts/n2_quality.py'
    spec = importlib.util.spec_from_file_location('n2_quality_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'quality.json').write_text('[{"id":"x","messages":[],"answer":"#### 1"}]')
    (data / 'plan.json').write_text(
        json.dumps({'files': {'quality': file_hash(data / 'quality.json')}})
    )
    (tmp_path / 'token').write_text('dummy')
    config = {
        'manifest': {
            'profile': {
                'name': 'n2_length_controlled_v1',
                'dtype': 'bfloat16',
                'quantization': None,
            },
            'artifact_digest': 'a' * 64,
        },
        'expected_artifact_digest': 'a' * 64,
        'endpoint': 'https://unused',
        'token_file': str(tmp_path / 'token'),
        'tls': {},
    }
    (tmp_path / 'engine.json').write_text(json.dumps(config))
    monkeypatch.setitem(
        sys.modules,
        'transformers',
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: None)),
    )

    class Engine:
        def __init__(self, *a, **k):
            pass

        async def close(self):
            pass

    async def not_ready(*a):
        return False, []

    monkeypatch.setattr(m, 'RemoteEngine', Engine)
    monkeypatch.setattr(m, 'check_readiness', not_ready)
    with pytest.raises(RuntimeError, match='worker_not_ready'):
        await m.run(data, tmp_path / 'engine.json', tmp_path, tmp_path / 'out', 'bf16')
    assert json.loads((tmp_path / 'out/input.json').read_text())['case_file'] == str(
        (data / 'quality.json').resolve()
    )


def test_fixed_length_summary_keeps_failures_and_wrong_length_in_denominator():
    from serving_lab.inference_study import summarize_fixed

    good = {
        'protocol_valid': True,
        'finish_reason': 'length',
        'usage': {'completion_tokens': 128},
        'output_budget': 128,
        'ttft_scheduled': 0.2,
        'ttft_sent': 0.2,
        'e2e_scheduled': 2,
        'e2e_sent': 2,
        'shape': 'short-0',
        'send_lag': 0,
        'sent': True,
        'status': 200,
    }
    short = {**good, 'finish_reason': 'stop', 'usage': {'completion_tokens': 5}}
    rejected = {**good, 'protocol_valid': False, 'usage': None, 'status': 429}
    result = summarize_fixed([good, short, rejected], 3, 1)
    assert (
        result['planned'] == 3
        and result['fixed_length_valid'] == 1
        and result['slo_fraction'] == 1 / 3
    )


def test_quantizer_refuses_rehashed_evaluation_rows_and_changed_bytes(tmp_path):
    import json, runpy
    from pathlib import Path
    from serving_lab.lab import file_hash

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/n2_quantize.py'))
    panel = tmp_path / 'calibration.json'
    plan = tmp_path / 'plan.json'
    rows = [{'id': f'train-{i:04}', 'messages': []} for i in range(128)]
    panel.write_text(json.dumps(rows))
    plan.write_text(json.dumps({'files': {'calibration': file_hash(panel)}}))
    assert len(module['calibration_rows'](plan, panel)) == 128
    rows[0]['id'] = 'test-0000'
    panel.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='冻结面板'):
        module['calibration_rows'](plan, panel)
    plan.write_text(json.dumps({'files': {'calibration': file_hash(panel)}}))
    with pytest.raises(ValueError, match='train'):
        module['calibration_rows'](plan, panel)


def test_benchmark_audit_rejects_response_splicing_and_time_reversal():
    import json, runpy
    from pathlib import Path

    replay = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / 'scripts/n2_benchmark_audit.py')
    )['replay_stream']
    request = {'expected_version': 'v', 'input_tokens': 1, 'output_budget': 2}

    def event(identifier, elapsed):
        return {
            'elapsed': elapsed,
            'data': json.dumps({'id': identifier, 'model': 'v', 'choices': []}),
        }

    with pytest.raises(ValueError, match='不同响应ID'):
        replay({'events': [event('a', 0.1), event('b', 0.2)], 'protocol_valid': False}, request)
    with pytest.raises(ValueError, match='时间发生倒退'):
        replay({'events': [event('a', 0.2), event('a', 0.1)], 'protocol_valid': False}, request)


def test_full_quality_gate_rejects_smoke_and_variant_mislabel():
    from serving_lab.inference_study import (
        require_quality_proof,
        require_variant,
        QUALITY_PLAN_SHA256,
    )

    manifest = {
        'artifact_digest': 'a' * 64,
        'profile_digest': 'p' * 64,
        'profile': {'dtype': 'bfloat16', 'quantization': None},
    }
    proof = {
        'raw_verified': True,
        'gate_passed': True,
        'smoke': False,
        'cases': 256,
        'correct': 233,
        'score': 233 / 256,
        'artifact_digest': 'a' * 64,
        'profile_digest': 'p' * 64,
        'variant': 'bf16',
        'plan_sha256': QUALITY_PLAN_SHA256,
    }
    require_quality_proof(proof, manifest)
    with pytest.raises(ValueError, match='完整'):
        require_quality_proof(
            {**proof, 'smoke': True, 'cases': 4, 'correct': 4, 'score': 1.0}, manifest
        )
    quant = {
        'quant_method': 'compressed-tensors',
        'config_groups': {'g': {'weights': {'num_bits': 4, 'group_size': 128}}},
    }
    with pytest.raises(ValueError, match='实际量化'):
        require_variant(
            {**manifest, 'profile': {'dtype': 'bfloat16', 'quantization': quant}}, 'bf16'
        )


def test_awq_baseline_requires_full_audited_bf16_with_same_panel():
    from serving_lab.inference_study import require_baseline, QUALITY_PLAN_SHA256

    proof = {
        'raw_verified': True,
        'gate_passed': True,
        'smoke': False,
        'cases': 256,
        'correct': 233,
        'score': 233 / 256,
        'variant': 'bf16',
        'plan_sha256': QUALITY_PLAN_SHA256,
        'artifact_digest': 'a' * 64,
    }
    require_baseline(proof, QUALITY_PLAN_SHA256)
    for bad in (
        {**proof, 'raw_verified': False},
        {**proof, 'variant': 'awq'},
        {**proof, 'plan_sha256': 'x' * 64},
    ):
        with pytest.raises(ValueError):
            require_baseline(bad, QUALITY_PLAN_SHA256)


def test_quant_metadata_compatibility_refuses_nondefault_dtype():
    import runpy, copy
    from pathlib import Path

    fn = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/n2_quant_compat.py'))[
        'compatible_config'
    ]
    original = {
        'quantization_config': {
            'quant_method': 'compressed-tensors',
            'version': '0.13.0',
            'format': 'pack-quantized',
            'config_groups': {
                'g': {
                    'weights': {
                        'num_bits': 4,
                        'group_size': 128,
                        'symmetric': False,
                        'type': 'int',
                        'dynamic': False,
                        'scale_dtype': None,
                        'zp_dtype': 'torch.int8',
                    }
                }
            },
        }
    }
    value, removed = fn(original)
    assert (
        original['quantization_config']['config_groups']['g']['weights']['zp_dtype'] == 'torch.int8'
    )
    assert 'zp_dtype' not in value['quantization_config']['config_groups']['g']['weights']
    bad = copy.deepcopy(original)
    bad['quantization_config']['config_groups']['g']['weights']['scale_dtype'] = 'torch.float16'
    with pytest.raises(ValueError, match='dtype'):
        fn(bad)


def test_rejected_quantization_diagnostic_does_not_grant_quality():
    from serving_lab.inference_study import require_quality_proof, QUALITY_PLAN_SHA256

    q = {
        'quant_method': 'compressed-tensors',
        'config_groups': {'g': {'weights': {'num_bits': 4, 'group_size': 128}}},
    }
    m = {
        'artifact_digest': 'a',
        'profile_digest': 'p',
        'profile': {'dtype': 'bfloat16', 'quantization': q},
    }
    proof = {
        'raw_verified': True,
        'gate_passed': False,
        'smoke': False,
        'cases': 256,
        'correct': 223,
        'score': 223 / 256,
        'artifact_digest': 'a',
        'profile_digest': 'p',
        'variant': 'awq',
        'plan_sha256': QUALITY_PLAN_SHA256,
    }
    with pytest.raises(ValueError):
        require_quality_proof(proof, m)
    require_quality_proof(proof, m, diagnostic_rejected=True)
    assert proof['gate_passed'] is False
    with pytest.raises(ValueError):
        require_quality_proof({**proof, 'smoke': True}, m, diagnostic_rejected=True)


def test_benchmark_duration_includes_drain_and_shape_sequence_is_fixed():
    import runpy
    from pathlib import Path

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / 'scripts/n2_benchmark_audit.py')
    )
    raw = {
        'elapsed_seconds': 24,
        'started_monotonic': 100,
        'ended_monotonic': 140,
        'rows': [{'end_offset': 40, 'send_offset': 23, 'e2e_sent': 17}],
    }
    with pytest.raises(ValueError, match='窗口分母'):
        module['verify_duration'](raw, 1)
    module['verify_duration']({**raw, 'elapsed_seconds': 40}, 1)
    assert module['expected_samples']('mixed', 4) == [
        'prefill-0',
        'decode-0',
        'prefill-1',
        'decode-1',
    ]
    assert module['expected_samples']('prefill', 9) == [f'prefill-{i % 8}' for i in range(9)]
