"""集群对照保持同一原生wire，非法动作不能混入应用口径。"""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_cluster.py'
    spec = importlib.util.spec_from_file_location('benchmark_cluster_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_native_requests_use_frozen_wire_and_fixed_two_endpoint_assignment():
    benchmark = module()
    sample = {
        'id': 's',
        'version': 'v',
        'profile_digest': 'p',
        'input_token_ids': [1, 2],
        'body': {'model': 'active', 'max_tokens': 64, 'action_choices': ['look']},
        'native_body': {
            'model': 'v',
            'max_tokens': 64,
            'seed': 2,
            'structured_outputs': {'choice': ['look']},
        },
    }
    plan = {'id': 'run', 'samples': [sample], 'endpoints': {'native': ['https://a', 'https://b']}}
    spec = {'path': 'native', 'name': 'w', 'sample_ids': ['s', 's', 's']}
    requests = benchmark.requests_for(plan, spec)
    assert [request['endpoint'] for request in requests] == ['https://a', 'https://b', 'https://a']
    assert requests[0]['body'] == {**sample['native_body'], 'stream': True}
    assert len({request['headers']['X-Request-Id'] for request in requests}) == 3


def test_invalid_action_preserves_transport_success_and_failure_denominator():
    benchmark = module()
    samples = [{'id': 's', 'body': {'action_choices': ['look']}}]
    rows = [
        {'shape': 's', 'protocol_valid': True, 'content': 'not a command'},
        {'shape': 's', 'protocol_valid': False, 'status': 429},
    ]
    benchmark.classify_actions(rows, samples, 'native')
    assert rows[0]['transport_valid'] and not rows[0]['action_valid']
    assert not rows[1]['transport_valid'] and not rows[1]['action_valid']
    assert len(rows) == 2
