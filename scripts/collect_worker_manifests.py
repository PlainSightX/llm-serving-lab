"""控制者启动前通过mTLS取得固定两个worker的当前身份，再核应用证据。"""

import argparse
import json
from pathlib import Path
import time
import uuid

import httpx

from serving_lab.application_release import load_policy
from serving_lab.lab import canonical
from serving_lab.tls import client_context
from serving_lab.worker_protocol import RemoteEngine


def collect(template, output):
    config = json.loads(Path(template).read_text(encoding='utf-8'))
    policy = load_policy(
        config['application_release']['bundle'], config['application_release']['sha256']
    )
    directory = Path(output)
    attempt = directory / ('attempt-' + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    engines, parameters = {}, {}
    for item in config['replicas']:
        token = Path(item['token_file']).read_text().strip()
        deadline = time.monotonic() + 300
        with httpx.Client(verify=client_context(item['tls']), trust_env=False, timeout=5) as client:
            while True:
                try:
                    response = client.get(
                        item['endpoint'] + '/identity', headers={'Authorization': 'Bearer ' + token}
                    )
                    response.raise_for_status()
                    value = response.json()
                    if value['ready']:
                        break
                except (httpx.HTTPError, KeyError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('等待worker就绪超时: ' + item['id'])
                time.sleep(1)
        manifest = value['manifest']
        engine = RemoteEngine(item['endpoint'], token, manifest, tls=item['tls'])
        engines[item['id']] = engine
        identity = {'digest': engine.parameter_digest}
        if engine.version_id in parameters and parameters[engine.version_id] != identity:
            raise ValueError('同版本的参数身份不一致')
        parameters[engine.version_id] = identity
        path = attempt / (item['id'] + '.json')
        path.write_text(canonical(manifest), encoding='utf-8')
        item['manifest'] = str(path)
    try:
        applications = policy.bind(engines, parameters)
        if not all(value['passed'] for value in applications.values()):
            raise ValueError('目标应用worker未取得质量资格')
        if len({engine.identity['worker_instance_id'] for engine in engines.values()}) != len(
            engines
        ):
            raise ValueError('两个副本引用了同一worker实例')
        (directory / 'gateway.json').write_text(canonical(config), encoding='utf-8')
    finally:
        import asyncio

        async def close():
            for engine in engines.values():
                await engine.close()

        asyncio.run(close())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='/config/gateway-template.json')
    parser.add_argument('--output', default='/bootstrap')
    args = parser.parse_args()
    collect(args.template, args.output)
