"""复用既有open-loop计量器，通过mTLS发送冻结的应用请求。"""

import argparse
import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import uuid

from serving_lab.benchmark import window, distribution
from serving_lab.execution import write_once
from serving_lab.lab import file_hash


async def run(endpoint, workload, certificates, output, count, rate, budget):
    if (
        not endpoint.startswith('https://')
        or not 2 <= count <= budget <= 300
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError('需要HTTPS、正速率以及2<=请求数<=预算<=300')
    specification = json.loads(Path(workload).read_text(encoding='utf-8'))
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(certificates)
    tls = {
        'ca': str(root / 'ca.pem'),
        'cert': str(root / 'loadgen.pem'),
        'key': str(root / 'loadgen.key'),
    }
    requests = []
    for index in range(count):
        sample = specification['samples'][index % len(specification['samples'])]
        requests.append(
            {
                'id': uuid.uuid4().hex,
                'shape': sample['id'],
                'path': 'gateway',
                'expected_version': sample['version'],
                'expected_profile': sample['profile_digest'],
                'input_tokens': len(sample['input_token_ids']),
                'output_budget': 64,
                'body': {**sample['body'], 'stream': True},
                'headers': {},
            }
        )
    write_once(
        directory / 'input.json',
        {
            'endpoint': endpoint,
            'workload_sha256': file_hash(workload),
            'reserved_calls': count,
            'budget': budget,
            'rate': rate,
            'requests': requests,
        },
    )
    result = await window(
        endpoint,
        requests,
        [index / rate for index in range(count)],
        {'client_timeout_seconds': 100, 'late_drop_seconds': 1, 'tls': tls},
    )
    for index, row in enumerate(result['rows']):
        sample = specification['samples'][index % len(specification['samples'])]
        if row['protocol_valid']:
            headers = row['response_headers']
            bound = (
                headers.get('x-application-evidence') == sample['application_evidence']
                and headers.get('x-qualification-scope') == 'application_replica_pool_v1'
                and headers.get('x-replica-id')
                and headers.get('x-worker-generation')
                and row['content'] in sample['body']['action_choices']
            )
            if not bound:
                row.update(protocol_valid=False, error='application_binding_or_choice_mismatch')
    write_once(directory / 'observations.json', result)
    valid = [row for row in result['rows'] if row['protocol_valid']]
    summary = {
        'arrivals': len(result['rows']),
        'valid': len(valid),
        'status_counts': dict(Counter(str(row['status']) for row in result['rows'])),
        'replicas': dict(Counter(row['response_headers'].get('x-replica-id') for row in valid)),
        'ttft_seconds': distribution([row['ttft_scheduled'] for row in valid]),
        'e2e_seconds': distribution([row['e2e_scheduled'] for row in valid]),
        'send_lag_seconds': distribution([row['send_lag'] for row in result['rows']]),
        'scope': 'bounded application load observations',
        'reserved_calls': count,
    }
    write_once(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--certificates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', type=int, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--budget', type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run(
                    args.endpoint,
                    args.workload,
                    args.certificates,
                    args.output,
                    args.requests,
                    args.rate,
                    args.budget,
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
