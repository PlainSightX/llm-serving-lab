import json
import struct

import pytest

from test_adaptive import task


def fixture(directory, value=0.0):
    directory.mkdir()
    (directory / 'adapter_config.json').write_text(
        json.dumps({'peft_type': 'LORA', 'base_model_name_or_path': 'fixture-only', 'r': 1})
    )
    header = json.dumps(
        {'fixture': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
    ).encode()
    (directory / 'adapter_model.safetensors').write_bytes(
        struct.pack('<Q', len(header)) + header + struct.pack('<f', value)
    )
    return directory


def test_gate_and_rollback_are_bound_to_bytes(tmp_path):
    from serving_lab.lab import Lab, LabError

    a = fixture(tmp_path / 'a')
    b = fixture(tmp_path / 'b', 1)
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout', 1)])
        for model, path in [('a', a), ('b', b)]:
            lab.register(model, path, fixture=True)
            lab.record(model + '-attempt', 'h', model, {'actions': ['check_units', 'compare']})
            lab.evaluate(model + '-eval', model, ['h'])
        lab.promote('a-eval')
        with pytest.raises(LabError):
            lab.promote('b-eval')
        lab.promote('b-eval', expected_active='a')
        assert lab.rollback('a', expected_active='b') == 'a'
        with pytest.raises(LabError):
            lab.promote('b-eval', expected_active='a', level='research')
        (b / 'adapter_config.json').write_text('{}')
        with pytest.raises(LabError):
            lab.promote('b-eval', expected_active='a')


def test_missing_failed_panel_and_rerun_cannot_pass(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout', 1)])
        lab.register('a', fixture(tmp_path / 'a'), fixture=True)
        with pytest.raises(LabError):
            lab.evaluate('missing', 'a', ['h'])
        lab.record('first', 'h', 'a', {})
        lab.record('second', 'h', 'a', {'actions': ['check_units', 'compare']})
        result = lab.evaluate('e', 'a', ['h'])
        assert not result['passed']
        with pytest.raises(LabError):
            lab.promote('e')


def test_fixture_cannot_register_as_trained_model(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path / 'lab') as lab:
        with pytest.raises(LabError):
            lab.register('fake', fixture(tmp_path / 'a'))


def test_partial_panel_and_retroactive_identity_rejected(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([task('h', 'holdout', 1), task('h2', 'holdout', 2)])
        lab.record('pre', 'h', 'a', {'actions': ['check_units', 'compare']})
        lab.register('a', fixture(tmp_path / 'a'), fixture=True)
        lab.record('post', 'h2', 'a', {})
        with pytest.raises(LabError):
            lab.evaluate('subset', 'a', ['h'])
        with pytest.raises(LabError):
            lab.evaluate('retro', 'a', ['h', 'h2'])


def test_same_parameters_cannot_rerun_under_new_alias(tmp_path):
    from serving_lab.lab import Lab, LabError

    a = fixture(tmp_path / 'a')
    b = fixture(tmp_path / 'b')
    (b / 'notes.txt').write_text('非参数元数据不同不能创造新候选')
    with Lab(tmp_path / 'lab') as lab:
        lab.register('first', a, fixture=True)
        with pytest.raises(LabError):
            lab.register('alias', a, fixture=True)
        with pytest.raises(LabError):
            lab.register('metadata_alias', b, fixture=True)
