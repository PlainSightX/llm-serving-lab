"""派生配对表不能靠自洽值伪装成三个等输入的真实调用。"""

import copy
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_application_release as audit


def fixture():
    pairs = []
    direct = {}
    reservations = {}
    clients = {}
    wires = {}
    for i in range(1, 4):
        d = {
            'arrival_id': f'd{i}',
            'attempt_id': f'a{i}',
            'content': 'look',
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
        }
        g = {'headers': {'x-arrival-id': f'g{i}'}, 'content': 'look', 'usage': d['usage']}
        payload = {
            'model': 'c0',
            'messages': [{'role': 'user', 'content': f'state-{i}'}],
            'structured_outputs': {'choice': ['look']},
            'stream': False,
        }
        direct[d['arrival_id']] = d
        reservations[d['arrival_id']] = {
            'arrival_id': d['arrival_id'],
            'attempt_id': d['attempt_id'],
            'metadata': {'pair': i},
            'wire': payload,
            'profile_digest': 'profile',
            'input_token_ids': [i],
        }
        clients[f'g{i}'] = ({}, g)
        wires[f'g{i}'] = {
            'payload': {**payload, 'stream': i == 3},
            'profile_digest': 'profile',
            'input_token_ids': [i],
        }
        if i == 3:
            wires[f'g{i}']['payload']['stream_options'] = {'include_usage': True}
        pairs.append({'pair': i, 'direct': d, 'gateway': g})
    return pairs, direct, reservations, clients, wires


def test_pair_check_binds_raw_ids_and_full_inputs():
    assert hasattr(audit, 'verify_pairs')
    args = fixture()
    audit.verify_pairs(*args)
    duplicate = copy.deepcopy(args)
    duplicate[0][1] = duplicate[0][0]
    with pytest.raises(ValueError):
        audit.verify_pairs(*duplicate)
    swapped = copy.deepcopy(args)
    swapped[0][0]['gateway'], swapped[0][1]['gateway'] = (
        swapped[0][1]['gateway'],
        swapped[0][0]['gateway'],
    )
    with pytest.raises(ValueError):
        audit.verify_pairs(*swapped)
    forged = list(copy.deepcopy(args))
    forged[0] = copy.deepcopy(forged[0])
    forged[0][0]['direct']['content'] = forged[0][0]['gateway']['content'] = 'forged'
    with pytest.raises(ValueError):
        audit.verify_pairs(*forged)


def test_stream_pair_allows_only_absent_null_optional_usage_detail():
    args = fixture()
    # vLLM真实非流式为null，流式省略；两侧三个token计数与输出仍需相同。
    args[0][2]['direct']['usage'] = {**args[0][2]['direct']['usage'], 'prompt_tokens_details': None}
    audit.verify_pairs(*args)
    mismatch = copy.deepcopy(args)
    mismatch[0][2]['gateway']['usage']['completion_tokens'] += 1
    with pytest.raises(ValueError):
        audit.verify_pairs(*mismatch)
    nonempty = copy.deepcopy(args)
    nonempty[0][2]['direct']['usage']['prompt_tokens_details'] = {'cached_tokens': 1}
    with pytest.raises(ValueError):
        audit.verify_pairs(*nonempty)


def test_support_binding_checks_original_task_image_and_file_bytes(tmp_path):
    from serving_lab.lab import file_hash

    assert hasattr(audit, 'check_support_source')
    tid = 'q1_support_01'
    files = {}
    for leaf in ('game.tw-pddl', 'traj_data.json'):
        name = f'games/{tid}/{leaf}'
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture')
        files[name] = file_hash(p)
    task = {'task_id': tid, 'role': 'support', 'files': files}
    record = {'task_id': tid, 'role': 'support'}
    env = {'task_id': tid}
    launch = {'image': 'pinned-image', 'command': ['create', 'pinned-image', '/runner.py']}
    audit.check_support_source(record, env, launch, task, tmp_path, 'pinned-image')
    with pytest.raises(ValueError):
        audit.check_support_source(
            record, {'task_id': 'q1_support_02'}, launch, task, tmp_path, 'pinned-image'
        )
    with pytest.raises(ValueError):
        audit.check_support_source(
            record, env, {**launch, 'image': 'other'}, task, tmp_path, 'pinned-image'
        )
    (tmp_path / f'games/{tid}/game.tw-pddl').write_text('changed')
    with pytest.raises(ValueError):
        audit.check_support_source(record, env, launch, task, tmp_path, 'pinned-image')


def test_failed_run_budget_carries_forward_and_binds_evidence(tmp_path):
    import probe_application_release as runner

    assert hasattr(runner, 'prior_failures')
    failed = tmp_path / 'a'
    failed.mkdir()

    def write(name, value):
        (failed / name).write_text(json.dumps(value))

    write('input.json', {'evidence_sha256': 'evidence', 'generation_cap': 680})
    write('failure.json', {'generation_reservations': 2})
    write('cleanup.json', {'errors': [], 'workers': [], 'processes': []})
    (failed / 'reservations.jsonl').write_text('{"reservation":2,"total":2}\n')
    records = runner.prior_failures([failed], 'evidence')
    assert records[0]['reservations'] == 2 and 'failure.json' in records[0]['files']
    with pytest.raises(ValueError):
        runner.prior_failures([failed, failed], 'evidence')
    with pytest.raises(ValueError):
        runner.prior_failures([failed], 'another-evidence')
    write('failure.json', {'generation_reservations': 1})
    with pytest.raises(ValueError):
        runner.prior_failures([failed], 'evidence')
    write('failure.json', {'generation_reservations': 2})
    write(
        'input.json',
        {
            'evidence_sha256': 'evidence',
            'generation_cap': 680,
            'prior_failed_runs': [{'path': str(tmp_path / 'omitted')}],
        },
    )
    with pytest.raises(ValueError):
        runner.prior_failures([failed], 'evidence')
    write('input.json', {'evidence_sha256': 'evidence', 'generation_cap': 680})
    write('summary.json', {'verified': True})
    with pytest.raises(ValueError):
        runner.prior_failures([failed], 'evidence')
