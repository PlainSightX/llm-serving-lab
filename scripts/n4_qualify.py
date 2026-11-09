"""恢复训练工件通过既有真实worker工程门；应用资格继续为未评价。"""

import argparse
import asyncio
import json
from pathlib import Path

from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.qualification import qualify_engine
from serving_lab.runtime import RuntimeStore
from serving_lab.worker_protocol import RemoteEngine


async def main(args):
    state = Path(args.state)
    secrets = Path(args.secrets)
    out = Path(args.output)
    pointer = json.loads((state / 'current.json').read_text())
    manifest = json.loads(Path(pointer['manifest']).read_text())
    if (
        manifest['artifact_digest'] != args.artifact
        or manifest['parameter_digest'] != args.parameters
    ):
        raise ValueError('实际worker没有装载已核恢复工件')
    if manifest['profile']['name'] != 'greedy_non_thinking_v1':
        raise ValueError('工程消费使用固定原生停止profile')
    engine = RemoteEngine(
        'https://localhost:8443',
        (secrets / 'worker-token').read_text().strip(),
        manifest,
        tls={
            'ca': str(secrets / 'ca.pem'),
            'cert': str(secrets / 'tls.pem'),
            'key': str(secrets / 'tls.key'),
        },
    )
    out.mkdir(parents=True, exist_ok=False)
    store = RuntimeStore(out / 'runtime')
    try:
        panel = json.loads(Path(args.panel).read_text())
        write_once(
            out / 'input.json',
            {
                'manifest': manifest,
                'panel_sha256': file_hash(args.panel),
                'source_sha256': file_hash(__file__),
                'artifact_digest': args.artifact,
                'parameter_digest': args.parameters,
                'application_qualified': False,
            },
        )
        result = await qualify_engine(
            engine,
            panel,
            out / 'qualification',
            store=store,
            application_result={
                'status': 'not_evaluated',
                'scope': 'n4_full_parameter_training',
                'meaning': 'no C0 application certificate inherited',
            },
        )
        if not result['passed']:
            raise ValueError('工程资格门未通过')
        write_once(
            out / 'summary.json',
            {
                'transport_passed': True,
                'qualified_cases': result['passed_cases'],
                'artifact_digest': args.artifact,
                'parameter_digest': args.parameters,
                'application_qualified': False,
                'publication_status': 'not_published',
                'scope': 'existing four-case real transport qualification only',
            },
        )
        print(json.dumps(result))
    finally:
        await engine.close()
        store.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['state', 'secrets', 'output', 'panel', 'artifact', 'parameters']:
        p.add_argument('--' + name, required=True)
    asyncio.run(main(p.parse_args()))
