"""固定长度测量显式隔离于原应用profile，不能任意覆盖解码配置。"""

import json
from types import SimpleNamespace
import pytest
from serving_lab.profile import make_profile
from serving_lab.online import Service, ServiceLimits, Rejected


def profile(tmp_path, policy):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{"eos_token_id":1}')
    (tmp_path / 'launch.json').write_text(
        json.dumps(
            {
                'command': ['--generation-config', 'vllm'],
                'versions': {'vllm': '0.11.2'},
                'config': {'dtype': 'bfloat16', 'max_model_len': 4096},
            }
        )
    )
    binding = SimpleNamespace(
        manifest_path=tmp_path / 'binding.json',
        model_path=model,
        artifact_digest='a' * 64,
        served_model='n2-model',
    )
    tokenizer = SimpleNamespace(get_chat_template=lambda: 'template', eos_token_id=1)
    return make_profile(binding, tokenizer, policy)


def test_fixed_length_is_only_an_explicit_bounded_measurement_override(tmp_path):
    p = profile(tmp_path, {'name': 'n2_length_controlled_v1', 'seed': 0})
    body = {
        'model': 'n2-model',
        'messages': [{'role': 'user', 'content': 'x'}],
        'max_tokens': 512,
        'stream': True,
    }
    assert p.payload(body)['ignore_eos'] is False
    assert p.payload({**body, 'ignore_eos': True})['ignore_eos'] is True
    with pytest.raises(ValueError):
        p.payload({**body, 'ignore_eos': 'true'})
    with pytest.raises(ValueError):
        p.payload({**body, 'max_tokens': 513})
    with pytest.raises(ValueError):
        p.payload({**body, 'temperature': 1})


def test_old_profile_does_not_gain_the_benchmark_override(tmp_path):
    p = profile(tmp_path, {'name': 'q2_public_choice_v1', 'seed': 20261103})
    with pytest.raises(ValueError):
        p.payload({'model': 'n2-model', 'messages': [], 'max_tokens': 64, 'ignore_eos': True})
    assert p.document['max_tokens'] == 64


def test_gateway_keeps_the_same_validated_flag_on_its_wire(tmp_path):
    p = profile(tmp_path, {'name': 'n2_length_controlled_v1', 'seed': 0})
    engine = SimpleNamespace(version_id='n2-model', profile=p)
    service = Service(
        engine,
        tmp_path / 'runtime',
        ServiceLimits(max_tokens=512, max_input_tokens=3500, context_tokens=4096),
    )
    try:
        body = {
            'model': 'n2-model',
            'messages': [{'role': 'user', 'content': 'x'}],
            'max_tokens': 512,
            'ignore_eos': True,
        }
        assert service.validate(body)['ignore_eos'] is True
        with pytest.raises(Rejected):
            service.validate({**body, 'ignore_eos': 1})
    finally:
        service.store.close()
