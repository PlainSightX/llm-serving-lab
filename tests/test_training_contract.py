import pytest
from serving_lab.demo import run_demo
from serving_lab.lab import LabError
from serving_lab.train import validate_export


def test_research_rejects_fixture_and_modified_data(tmp_path):
    run_demo(tmp_path / 'run')
    data = tmp_path / 'run/training_export'
    assert len(validate_export(data, 'model_smoke')[1]) == 8
    with pytest.raises(LabError):
        validate_export(data, 'research')
    with (data / 'train.jsonl').open('a', encoding='utf-8') as f:
        f.write('{}\n')
    with pytest.raises(LabError):
        validate_export(data, 'model_smoke')
