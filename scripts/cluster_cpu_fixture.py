"""本地Kubernetes网络演练专用CPU替身，只返回明确标记的运输层回执。"""

import argparse
import hmac
import json
from pathlib import Path
import uuid
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from serving_lab.tls import client_context, listener_options


def server(secrets):
    root = Path(secrets)
    token = (root / 'worker-token').read_text().strip()
    instance = uuid.uuid4().hex
    app = FastAPI()

    @app.get('/identity')
    def identity(request: Request):
        if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        return {'evidence_level': 'cpu_transport_fixture', 'instance': instance}

    uvicorn.run(
        app,
        **listener_options(
            '0.0.0.0',
            8443,
            ca=str(root / 'ca.pem'),
            cert=str(root / 'tls.pem'),
            key=str(root / 'tls.key'),
        ),
    )


def client(secrets, namespace):
    root = Path(secrets)
    material = {
        'ca': str(root / 'ca.pem'),
        'cert': str(root / 'client.pem'),
        'key': str(root / 'client.key'),
    }
    results = {}
    with httpx.Client(verify=client_context(material), trust_env=False, timeout=10) as connection:
        for slot in ('a', 'b'):
            endpoint = f'https://worker-{slot}.{namespace}.svc.cluster.local:8443/identity'
            token = (root / f'worker-{slot}-token').read_text().strip()
            # Pod Ready与Service转发表更新分属不同控制循环，先等只读接口可达。
            deadline = time.monotonic() + 30
            attempts = 0
            while True:
                attempts += 1
                try:
                    response = connection.get(
                        endpoint, headers={'Authorization': 'Bearer ' + token}
                    )
                    break
                except (httpx.ConnectError, httpx.ReadTimeout):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.5)
            response.raise_for_status()
            value = response.json()
            assert value['evidence_level'] == 'cpu_transport_fixture'
            assert (
                connection.get(
                    endpoint, headers={'Authorization': 'Bearer wrong-token'}
                ).status_code
                == 401
            )
            results[slot] = value
            results[slot]['readiness_attempts'] = attempts
    assert results['a']['instance'] != results['b']['instance']
    print(
        json.dumps(
            {'verified': True, 'evidence_level': 'cpu_transport_fixture', 'workers': results}
        ),
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['server', 'client'])
    parser.add_argument('--secrets', default='/secrets')
    parser.add_argument('--namespace', default='lsl-r5')
    args = parser.parse_args()
    if args.mode == 'server':
        server(args.secrets)
    else:
        client(args.secrets, args.namespace)
