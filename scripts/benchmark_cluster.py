"""消费冻结窗口，复用同一计量器记录集群各入口及公开动作合法性。"""

import argparse
import asyncio
import json
from pathlib import Path

from serving_lab.benchmark import summarize, window
from serving_lab.execution import write_once
from serving_lab.lab import file_hash


def requests_for(plan, specification):
    samples = {sample['id']: sample for sample in plan['samples']}
    path = specification['path']
    endpoints = plan['endpoints'][path]
    requests = []
    for index, sample_id in enumerate(specification['sample_ids']):
        sample = samples[sample_id]
        request_id = f"{plan['id']}-{specification['name']}-{index:04}"
        body = sample['body'] if path == 'gateway' else sample['native_body']
        requests.append(
            {
                'id': request_id,
                'shape': sample_id,
                'path': path,
                'endpoint': endpoints[index % len(endpoints)],
                'expected_version': sample['version'],
                'expected_profile': sample['profile_digest'],
                'input_tokens': len(sample['input_token_ids']),
                'output_budget': body['max_tokens'],
                'body': {**body, 'stream': True},
                'headers': {'X-Request-Id': request_id, 'Idempotency-Key': request_id},
            }
        )
    return requests


def classify_actions(rows, samples, path):
    """合法动作仅表示当次公开候选内的完整命令，任务正确性另行评价。"""
    indexed = {sample['id']: sample for sample in samples}
    for row in rows:
        row['transport_valid'] = row['protocol_valid']
        sample = indexed[row['shape']]
        valid = row['protocol_valid'] and row.get('content') in sample['body']['action_choices']
        if path == 'gateway' and valid:
            headers = row['response_headers']
            valid = (
                headers.get('x-application-evidence') == sample['application_evidence']
                and headers.get('x-qualification-scope') == 'application_replica_pool_v1'
                and bool(headers.get('x-replica-id'))
                and bool(headers.get('x-worker-generation'))
            )
        row['action_valid'] = bool(valid)
        row['protocol_valid'] = bool(valid)
        if row['transport_valid'] and not valid:
            row['action_error'] = 'public_choice_or_application_binding_mismatch'


async def run(plan_path, name, certificates, output):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    specification = next(item for item in plan['windows'] if item['name'] == name)
    requests = requests_for(plan, specification)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    certificate_root = Path(certificates)
    settings = {
        **plan['client'],
        'tls': {
            'ca': str(certificate_root / 'ca.pem'),
            'cert': str(certificate_root / 'loadgen.pem'),
            'key': str(certificate_root / 'loadgen.key'),
        },
    }
    write_once(
        directory / 'input.json',
        {
            'plan_sha256': file_hash(plan_path),
            'specification': specification,
            'requests': requests,
            'settings': settings,
        },
    )
    result = await window(requests[0]['endpoint'], requests, specification['offsets'], settings)
    arguments = {'interval': 1 / specification['rate'], 'lag_fraction': plan['lag_fraction']}
    protocol = summarize(result['rows'], result['elapsed_seconds'], plan['slo'], **arguments)
    classify_actions(result['rows'], plan['samples'], specification['path'])
    action = summarize(result['rows'], result['elapsed_seconds'], plan['slo'], **arguments)
    write_once(directory / 'observations.json', result)
    summary = {
        'name': name,
        'path': specification['path'],
        'rate': specification['rate'],
        'repeat': specification['repeat'],
        'partition': specification['partition'],
        'transport': protocol,
        'action': action,
        'target_met': action['schedule_valid']
        and action['slo_fraction'] >= plan['required_fraction'],
        'scope': 'frozen short-command engineering load windows',
    }
    write_once(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--certificates', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.plan, args.window, args.certificates, args.output))))
