"""原始观测决定达标与配对资格，存档结论不能覆盖重算结果。"""

import importlib.util
import json
from pathlib import Path

import pytest

from serving_lab.benchmark import summarize
from serving_lab.lab import file_hash


def load_reporter(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        'cluster_report_test', scripts / 'report_cluster_benchmark.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(root, reporter, *, rate=8, native_lag=0, native_ttft=0.05):
    sample = {
        'id': 's',
        'version': 'm',
        'profile_digest': 'p',
        'application_evidence': 'e',
        'input_token_ids': [1, 2],
        'body': {'max_tokens': 16, 'action_choices': ['look']},
        'native_body': {'model': 'm', 'max_tokens': 16},
    }
    windows = [
        {
            'name': 'w-' + path,
            'path': path,
            'repeat': 0,
            'rate': rate,
            'partition': 'main',
            'sample_ids': ['s'],
            'offsets': [0],
        }
        for path in ('native', 'gateway', 'upstream')
    ]
    plan = {
        'id': 'test',
        'samples': [sample],
        'windows': windows,
        'endpoints': {path: ['https://' + path] for path in ('native', 'gateway', 'upstream')},
        'client': {},
        'slo': {'ttft_seconds': 1, 'short_e2e_seconds': 2, 'long_e2e_seconds': 2},
        'lag_fraction': 0.1,
        'required_fraction': 0.99,
        'formal_requests': 3,
    }

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    plan_path = root / 'plan-v2.json'
    save(plan_path, plan)
    for specification in windows:
        request = reporter.requests_for(plan, specification)[0]
        native = specification['path'] == 'native'
        lag, ttft = (native_lag, native_ttft) if native else (0, 0.05)
        done = ttft + 0.05
        usage = {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}
        event = {
            'model': 'm',
            'choices': [{'delta': {'content': 'look'}, 'finish_reason': 'stop'}],
            'usage': usage,
        }
        row = {
            'id': request['id'],
            'shape': 's',
            'scheduled_offset': 0,
            'input_tokens': 2,
            'output_budget': 16,
            'status': 200,
            'sent': True,
            'send_lag': lag,
            'events': [
                {'data': json.dumps(event), 'elapsed': ttft},
                {'data': '[DONE]', 'elapsed': done},
            ],
            'transport_valid': True,
            'protocol_valid': True,
            'content': 'look',
            'usage': usage,
            'ttft_sent': ttft,
            'e2e_sent': done,
            'ttft_scheduled': lag + ttft,
            'e2e_scheduled': lag + done,
            'finish_reason': 'stop',
            'response_headers': {
                'x-application-evidence': 'e',
                'x-qualification-scope': 'application_replica_pool_v1',
                'x-replica-id': 'a',
                'x-worker-generation': 'g',
            },
        }
        rows = [row]
        protocol = summarize(rows, 3, plan['slo'], interval=1 / rate, lag_fraction=0.1)
        reporter.classify_actions(rows, [sample], specification['path'])
        action = summarize(rows, 3, plan['slo'], interval=1 / rate, lag_fraction=0.1)
        summary = {
            key: specification[key] for key in ('name', 'path', 'rate', 'repeat', 'partition')
        }
        summary.update(
            transport=protocol,
            action=action,
            target_met=action['schedule_valid'] and action['slo_fraction'] >= 0.99,
        )
        directory = root / 'evidence/loadgen' / specification['name']
        save(
            directory / 'input.json',
            {
                'plan_sha256': file_hash(plan_path),
                'specification': specification,
                'requests': [request],
            },
        )
        save(directory / 'observations.json', {'rows': rows, 'elapsed_seconds': 3})
        save(directory / 'summary.json', summary)
    return plan


def test_report_rejects_forged_target_met(tmp_path, monkeypatch):
    reporter = load_reporter(monkeypatch)
    make_run(tmp_path, reporter, native_ttft=2)
    path = tmp_path / 'evidence/loadgen/w-native/summary.json'
    summary = json.loads(path.read_text())
    assert summary['action']['slo_met'] == 0
    summary['target_met'] = True
    path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        reporter.report(tmp_path, tmp_path / 'report.json')


def test_report_rejects_window_metadata_mismatch(tmp_path, monkeypatch):
    reporter = load_reporter(monkeypatch)
    make_run(tmp_path, reporter)
    path = tmp_path / 'evidence/loadgen/w-native/summary.json'
    summary = json.loads(path.read_text())
    summary['rate'] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        reporter.report(tmp_path, tmp_path / 'report.json')


def test_invalid_generator_cannot_support_paired_overhead(tmp_path, monkeypatch):
    reporter = load_reporter(monkeypatch)
    make_run(tmp_path, reporter, rate=1, native_lag=0.2)
    report = reporter.report(tmp_path, tmp_path / 'report.json')
    assert not report['all_formal_schedules_valid']
    assert all(not item['attribution_qualified'] for item in report['paired'])
    assert all(
        item['interpretation'] == 'unqualified generator observation' for item in report['paired']
    )
    assert all(item['actual_length_matched'] == 1 for item in report['paired'])
