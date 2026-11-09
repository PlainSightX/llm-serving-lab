import json
import subprocess
import sys

import pytest

from serving_lab.lab import Lab, LabError
from test_adaptive import task


def base_dir(tmp_path):
    p = tmp_path / 'base'
    p.mkdir()
    (p / 'config.json').write_text('{"model_type":"test-only"}')
    (p / 'model.safetensors').write_bytes(b'unit-test-not-a-model')
    (p / 'tokenizer.json').write_text('{}')
    return p


def test_reservation_precedes_execution_and_survives_restart(tmp_path):
    from serving_lab.execution import begin_execution

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        lab.register_base('base', base_dir(tmp_path))
        request = begin_execution(lab, 'run1', 'h', 'base')
        assert 'expected' not in request['task']
        assert lab.db.execute('SELECT output FROM attempts').fetchone()[0] == 'null'
        with pytest.raises(LabError):
            lab.feedback('run1')
    with Lab(tmp_path / 'lab') as lab:
        assert lab.db.execute('SELECT state FROM executions').fetchone()[0] == 'unknown'
        with pytest.raises(LabError):
            begin_execution(lab, 'run2', 'h', 'base')


def test_added_base_file_invalidates_identity(tmp_path):
    p = base_dir(tmp_path)
    with Lab(tmp_path / 'lab') as lab:
        lab.register_base('base', p)
        (p / 'tokenizer_config.json').write_text('{}')
        with pytest.raises(LabError):
            lab.model('base')


def test_execution_request_tampering_is_detected(tmp_path):
    from serving_lab.execution import begin_execution, verify_execution

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'run1', 'h', 'base')
        path = tmp_path / 'lab/executions/run1/request.json'
        body = json.loads(path.read_text(encoding='utf-8'))
        body['task']['messages'][0]['content'] = 'modified'
        path.write_text(json.dumps(body), encoding='utf-8')
        with pytest.raises(LabError):
            verify_execution(lab, 'run1')


def complete_contract_fixture(lab, attempt_id, output, *, ended_eos=True):
    """只测试内部回执合同，临时文件不是项目真实模型执行证据。"""
    from serving_lab.execution import save_input, finish_execution
    from pathlib import Path
    from serving_lab.lab import file_hash
    import serving_lab

    folder = lab.root / 'executions' / attempt_id
    sources = {}
    for name in ('inference.py', 'execution.py', 'lab.py', 'business.py', 'memory.py'):
        destination = folder / ('source_' + name)
        destination.write_bytes(Path(serving_lab.__file__).with_name(name).read_bytes())
        sources[destination.name] = file_hash(destination)
    save_input(
        lab,
        attempt_id,
        'unit rendered prompt',
        [1, 2],
        {
            'source_files': sources,
            'tokenizer_eos_token_id': 3,
            'versions': {'unit': 'not_real_inference'},
        },
    )
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    return finish_execution(
        lab,
        attempt_id,
        json.dumps(output),
        [3] if ended_eos else [4],
        ended_eos=ended_eos,
        elapsed_seconds=0.01,
        runtime={
            'test_only': True,
            'loaded_model_id': request['model_id'],
            'loaded_artifact_digest': request['artifact_digest'],
            'loaded_parameter_identity': request['parameter_identity'],
            'prompt_tokens': 2,
            'completion_tokens': 1,
        },
    )


def test_real_receipt_gate_rejects_tamper_and_partial_output(tmp_path):
    from serving_lab.execution import begin_execution

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'run1', 'h', 'base')
        complete_contract_fixture(
            lab, 'run1', {'actions': ['check_units', 'compare']}, ended_eos=False
        )
        assert not lab.evaluate('e', 'base', ['h'])['passed']
        with pytest.raises(LabError):
            lab.promote('e', level='model_smoke')
        with pytest.raises(LabError):
            begin_execution(lab, 'run2', 'h', 'base')
        response = tmp_path / 'lab/executions/run1/response.json'
        result = json.loads(response.read_text())
        result['output_complete'] = True
        response.write_text(json.dumps(result))
        with pytest.raises(LabError):
            lab.evaluate('changed', 'base', ['h'])


def test_complete_gate_and_manual_evidence_do_not_mix(tmp_path):
    from serving_lab.execution import begin_execution

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'run1', 'h', 'base')
        complete_contract_fixture(lab, 'run1', {'actions': ['check_units', 'compare']})
        assert lab.evaluate('e', 'base', ['h'])['evidence_level'] == 'model_smoke'
        assert lab.promote('e', level='model_smoke') == 'base'
        with pytest.raises(LabError):
            lab.promote('e', expected_active='base', level='research')


def test_real_process_crash_retains_first_reservation(tmp_path):
    p = base_dir(tmp_path)
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        lab.register_base('base', p)
    code = 'from serving_lab.lab import Lab; from serving_lab.execution import begin_execution; import os,sys; lab=Lab(sys.argv[1]); begin_execution(lab,"crashed","h","base"); os._exit(17)'
    completed = subprocess.run(
        [sys.executable, '-c', code, str(tmp_path / 'lab')], capture_output=True
    )
    assert completed.returncode == 17
    with Lab(tmp_path / 'lab') as lab:
        state = lab.db.execute('SELECT state FROM executions').fetchone()[0]
        assert state == 'unknown'
        result = lab.evaluate('e', 'base', ['h'])
        assert not result['passed']
        with pytest.raises(LabError):
            lab.promote('e', level='model_smoke')


def test_nested_template_changes_identity(tmp_path):
    p = base_dir(tmp_path)
    with Lab(tmp_path / 'lab') as lab:
        lab.register_base('base', p)
        (p / 'chat_templates').mkdir()
        (p / 'chat_templates/default.jinja').write_text('changed prompt')
        with pytest.raises(LabError):
            lab.model('base')


def test_explicit_resume_preserves_unknown_and_rejects_new_settings(tmp_path):
    from serving_lab.execution import begin_execution, resume_existing, DEFAULT_GENERATION

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout', 1), task('h2', 'holdout', 2)])
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'first', 'h', 'base')
    with Lab(tmp_path / 'lab') as lab:
        resumed = resume_existing(lab, 'h', 'base', DEFAULT_GENERATION, 'frozen')
        assert resumed['state'] == 'unknown'
        assert resume_existing(lab, 'h2', 'base', DEFAULT_GENERATION, 'frozen') is None
        with pytest.raises(LabError):
            resume_existing(lab, 'h', 'base', {**DEFAULT_GENERATION, 'max_new_tokens': 2}, 'frozen')
        begin_execution(lab, 'next', 'h2', 'base')


def test_model_smoke_experience_remains_ineligible_for_research(tmp_path):
    from serving_lab.execution import begin_execution
    from serving_lab.train import validate_export

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('t', 'train')])
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'run1', 't', 'base')
        complete_contract_fixture(lab, 'run1', {'actions': ['check_units', 'compare']})
        lab.feedback('run1')
        manifest = lab.export_training(tmp_path / 'export', cutoff=1)
        assert manifest['evidence_level'] == 'model_smoke'
        assert len(validate_export(tmp_path / 'export', 'model_smoke')[1]) == 1
        with pytest.raises(LabError):
            validate_export(tmp_path / 'export', 'research')


def test_loaded_source_drift_is_rejected(monkeypatch):
    from serving_lab import inference

    inference.assert_source_stable()
    monkeypatch.setattr(
        inference, 'SOURCE_SNAPSHOT', {'source_inference.py': b'different loaded source'}
    )
    with pytest.raises(LabError):
        inference.assert_source_stable()


def test_public_record_cannot_claim_local_hf(tmp_path):
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        with pytest.raises(LabError):
            lab.record(
                'fake', 'h', 'base', {'actions': ['check_units', 'compare']}, actor='local_hf'
            )


def test_model_gate_rollback_rechecks_old_receipt(tmp_path):
    from serving_lab.execution import begin_execution

    base_a = base_dir(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    base_b = base_dir(other)
    (base_b / 'model.safetensors').write_bytes(b'different-unit-test-only-bytes')
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout')])
        for name, path in [('a', base_a), ('b', base_b)]:
            lab.register_base(name, path)
            begin_execution(lab, 'run_' + name, 'h', name)
            complete_contract_fixture(lab, 'run_' + name, {'actions': ['check_units', 'compare']})
            lab.evaluate('eval_' + name, name, ['h'])
        lab.promote('eval_a', level='model_smoke')
        lab.promote('eval_b', expected_active='a', level='model_smoke')
        assert lab.rollback('a', expected_active='b', level='model_smoke') == 'a'
        lab.promote('eval_b', expected_active='a', level='model_smoke')
        receipt = tmp_path / 'lab/executions/run_a/receipt.json'
        receipt.write_text('{}')
        with pytest.raises(LabError):
            lab.rollback('a', expected_active='b', level='model_smoke')
        assert lab.active() == 'b'
