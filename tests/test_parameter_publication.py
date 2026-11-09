"""跨参数发布的客户端判定不能只相信HTTP 200。"""

import importlib.util
from pathlib import Path
import pytest


def test_response_cannot_mix_version_profile_or_output():
    path = Path(__file__).resolve().parents[1] / 'scripts/probe_parameter_publication.py'
    assert path.exists()
    spec = importlib.util.spec_from_file_location('parameter_pub_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    receipt = {
        'status': 200,
        'version': 'c0',
        'profile': 'p0',
        'evidence': 'e0',
        'content': 'look',
        'finish_reason': 'stop',
        'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
        'route_generation': 1,
    }
    specs = {'c0': {'profile_digest': 'p0', 'evidence_digest': 'e0'}}
    payload = {'action_choices': ['look', 'inventory'], 'input_token_ids': [1, 2, 3]}
    m.validate_response(receipt, payload, specs, expected='c0')
    with pytest.raises(ValueError):
        m.validate_response({**receipt, 'version': 'c1'}, payload, specs, expected='c0')
    with pytest.raises(ValueError):
        m.validate_response({**receipt, 'profile': 'p1'}, payload, specs, expected='c0')
    with pytest.raises(ValueError):
        m.validate_response(
            {**receipt, 'content': 'not a current command'}, payload, specs, expected='c0'
        )


def test_pre_switch_snapshot_is_not_proof_of_queued_work_at_commit():
    import json

    path = Path(__file__).resolve().parents[1] / 'scripts/probe_parameter_publication.py'
    spec = importlib.util.spec_from_file_location('parameter_order_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def event(i, arrival, kind, body):
        return {'id': i, 'arrival_id': arrival, 'kind': kind, 'body': json.dumps(body)}

    events = [
        event(1, 'a', 'binding', {'version_id': 'c0', 'route_generation': 3}),
        event(
            2,
            None,
            'pool_route_changed',
            {
                'operation': 'promote',
                'previous': {'active': 'c0', 'canary': None, 'generation': 3},
                'current': {'active': 'c1', 'generation': 4},
            },
        ),
        event(3, 'a', 'dispatched', {}),
        event(4, 'a', 'terminal', {'state': 'completed'}),
    ]
    assert len(m.queued_cutover_proof(events, {'a'}, 'c0', 'c1', 4)) == 1
    before = [events[0], {**events[2], 'id': 2}, {**events[3], 'id': 3}, {**events[1], 'id': 4}]
    assert m.queued_cutover_proof(before, {'a'}, 'c0', 'c1', 4) == []
    changed = [dict(e) for e in events]
    changed[0] = event(1, 'a', 'binding', {'version_id': 'c1', 'route_generation': 3})
    assert m.queued_cutover_proof(changed, {'a'}, 'c0', 'c1', 4) == []


def test_stream_audit_rejects_a_chunk_from_another_model():
    import json

    path = Path(__file__).resolve().parents[1] / 'scripts/audit_parameter_publication.py'
    assert path.exists()
    spec = importlib.util.spec_from_file_location('pub_audit_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    event = {
        'id': 'chatcmpl-a',
        'model': 'c0',
        'choices': [{'delta': {'content': 'look'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
    }
    raw = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
    assert m.parse_response(raw, True, 'chatcmpl-a', 'c0')[0] == 'look'
    with pytest.raises(ValueError):
        m.parse_response(raw, True, 'chatcmpl-a', 'c1')


def test_optional_usage_details_do_not_change_equal_token_counts():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_parameter_publication.py'
    spec = importlib.util.spec_from_file_location('pub_usage_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = {
        'content': 'look',
        'finish_reason': 'stop',
        'usage': {
            'prompt_tokens': 3,
            'completion_tokens': 1,
            'total_tokens': 4,
            'prompt_tokens_details': None,
        },
    }
    b = {
        'content': 'look',
        'finish_reason': 'stop',
        'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
    }
    assert m.same_output(a, b)
    b['usage']['completion_tokens'] = 2
    assert not m.same_output(a, b)
