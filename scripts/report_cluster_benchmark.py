"""从冻结计划和原始SSE重算窗口，汇总有限负载结果与等长配对。"""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from serving_lab.benchmark import StreamObservation, distribution, summarize
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from benchmark_cluster import classify_actions, requests_for


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def report(root, output):
    root = Path(root)
    plan_path = root / 'plan-v2.json'
    plan = read(plan_path)
    windows = []
    groups = defaultdict(list)
    observations = {}
    for specification in plan['windows']:
        directory = root / 'evidence/loadgen' / specification['name']
        inputs = read(directory / 'input.json')
        data = read(directory / 'observations.json')
        stored = read(directory / 'summary.json')
        assert inputs['plan_sha256'] == file_hash(plan_path)
        assert inputs['specification'] == specification
        expected_requests = requests_for(plan, specification)
        assert inputs['requests'] == expected_requests
        assert len(data['rows']) == len(expected_requests)
        rows = copy.deepcopy(data['rows'])
        for index, row in enumerate(rows):
            request = expected_requests[index]
            assert row['id'] == request['id'] and row['shape'] == request['shape']
            assert row['scheduled_offset'] == specification['offsets'][index]
            assert row['input_tokens'] == request['input_tokens']
            assert row['output_budget'] == request['output_budget']
            parsed = StreamObservation(
                request['expected_version'], request['input_tokens'], request['output_budget']
            )
            for event in row.get('events', []):
                parsed.feed(event['data'], event['elapsed'])
            expected_transport = row['status'] == 200 and parsed.valid and not row.get('error')
            assert row['transport_valid'] == expected_transport
            if parsed.valid:
                assert row['content'] == parsed.content and row['usage'] == parsed.usage
                assert row['ttft_sent'] == parsed.first_content and row['e2e_sent'] == parsed.done
                assert row['ttft_scheduled'] == row['send_lag'] + parsed.first_content
                assert row['e2e_scheduled'] == row['send_lag'] + parsed.done
            row['protocol_valid'] = row['transport_valid']
        arguments = {'interval': 1 / specification['rate'], 'lag_fraction': plan['lag_fraction']}
        protocol = summarize(rows, data['elapsed_seconds'], plan['slo'], **arguments)
        assert protocol == stored['transport']
        classify_actions(rows, plan['samples'], specification['path'])
        action = summarize(rows, data['elapsed_seconds'], plan['slo'], **arguments)
        assert action == stored['action']
        for key in ('name', 'path', 'rate', 'repeat', 'partition'):
            assert stored[key] == specification[key], (specification['name'], key)
        target_met = (
            action['schedule_valid'] and action['slo_fraction'] >= plan['required_fraction']
        )
        assert stored['target_met'] == target_met, specification['name']
        verified = {**stored, 'transport': protocol, 'action': action, 'target_met': target_met}
        windows.append(verified)
        observations[specification['name']] = rows
        if specification['partition'] != 'warm':
            groups[
                (specification['partition'], specification['rate'], specification['path'])
            ].append((verified, rows))

    aggregates = []
    for (partition, rate, path), items in sorted(groups.items()):
        rows = [row for _, window_rows in items for row in window_rows]
        elapsed = sum(item['action']['elapsed_seconds'] for item, _ in items)
        summary = summarize(
            rows, elapsed, plan['slo'], interval=1 / rate, lag_fraction=plan['lag_fraction']
        )
        aggregates.append(
            {
                'partition': partition,
                'rate': rate,
                'path': path,
                'windows': len(items),
                'every_schedule_valid': all(item['action']['schedule_valid'] for item, _ in items),
                'every_target_met': all(item['target_met'] for item, _ in items),
                'planned_arrival_seconds': len(rows) / rate,
                'summary': summary,
                'repeat_slo_met': [item['action']['slo_met'] for item, _ in items],
                'repeat_ttft_median': [
                    item['action']['ttft_scheduled']['median'] for item, _ in items
                ],
            }
        )

    schedules = {item['name']: item['action']['schedule_valid'] for item in windows}
    paired = []
    for specification in plan['windows']:
        if specification['path'] != 'native' or specification['partition'] == 'warm':
            continue
        native = observations[specification['name']]
        for path in ('gateway', 'upstream'):
            other_name = specification['name'].removesuffix('native') + path
            other = observations[other_name]
            qualified = schedules[specification['name']] and schedules[other_name]
            interpretation = 'unqualified generator observation'
            if qualified:
                interpretation = (
                    'low-load matched extra wait'
                    if specification['rate'] == 1
                    else 'whole-system difference including admission and queue policy'
                )
            length_matched, output_equal, deltas = 0, 0, []
            for left, right in zip(native, other):
                assert (
                    left['shape'] == right['shape']
                    and left['scheduled_offset'] == right['scheduled_offset']
                )
                if not left['action_valid'] or not right['action_valid']:
                    continue
                if left['content'] == right['content']:
                    output_equal += 1
                if left['usage'] == right['usage']:
                    length_matched += 1
                    deltas.append(right['ttft_scheduled'] - left['ttft_scheduled'])
            paired.append(
                {
                    'partition': specification['partition'],
                    'repeat': specification['repeat'],
                    'rate': specification['rate'],
                    'path': path,
                    'planned_pairs': len(native),
                    'actual_length_matched': length_matched,
                    'exact_content_matched': output_equal,
                    'ttft_difference_seconds': distribution(deltas),
                    'native_schedule_valid': schedules[specification['name']],
                    'other_schedule_valid': schedules[other_name],
                    'attribution_qualified': qualified,
                    'interpretation': interpretation,
                }
            )

    formal = [item for item in windows if item['partition'] != 'warm']
    assert sum(item['action']['planned'] for item in formal) == plan['formal_requests']
    result = {
        'verified': True,
        'formal_requests': plan['formal_requests'],
        'formal_windows': len(formal),
        'warmup_requests': sum(
            item['action']['planned'] for item in windows if item['partition'] == 'warm'
        ),
        'aggregates': aggregates,
        'paired': paired,
        'all_formal_schedules_valid': all(item['action']['schedule_valid'] for item in formal),
        'all_core_target_windows_met': all(
            item['target_met'] for item in formal if item['rate'] == 8
        ),
        'scope': 'finite frozen public-command load; holdout is engineering-load holdout',
        'reporter_sha256': file_hash(__file__),
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ('aggregates', 'paired')},
            indent=2,
        )
    )
