"""控制者仅消费受控worker网络清单；参数文件和/proc属于worker本机。"""

import argparse
import json
from pathlib import Path
import uvicorn
from serving_lab.worker_protocol import RemoteEngine
from serving_lab.replica_api import create_replica_app
from serving_lab.online import ServiceLimits
from serving_lab.profile import application_history
from serving_lab.application_release import load_policy
from serving_lab.tls import listener_options


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding='utf-8'))
    listener = listener_options(
        config.get('host', '127.0.0.1'), config['port'], **config.get('tls', {})
    )
    engines = {}
    parameters = {}
    capacities = {}
    applications = {}
    for item in config['replicas']:
        rid = item['id']
        if rid in engines:
            raise ValueError('重复副本ID')
        manifest = json.loads(Path(item['manifest']).read_text(encoding='utf-8'))
        engine = RemoteEngine(
            item['endpoint'],
            Path(item['token_file']).read_text().strip(),
            manifest,
            tls=item.get('tls'),
        )
        engines[rid] = engine
        value = {'digest': engine.parameter_digest}
        if engine.version_id in parameters and parameters[engine.version_id] != value:
            raise ValueError('同版本副本参数不一致')
        parameters[engine.version_id] = value
        capacities[rid] = item['capacity']
        if not config.get('application_release'):
            applications[engine.version_id] = application_history(
                config['application_receipts'][engine.version_id],
                type(
                    'ArtifactReference',
                    (),
                    {
                        'version_id': engine.version_id,
                        'artifact_digest': engine.binding.artifact_digest,
                    },
                )(),
            )
    policy = None
    if config.get('application_release'):
        evidence = config['application_release']
        policy = load_policy(evidence['bundle'], evidence['sha256'])
        applications = policy.bind(engines, parameters)
    panel = json.loads(Path(config['panel']).read_text(encoding='utf-8'))
    app = create_replica_app(
        engines,
        config['runtime'],
        parameters,
        panel,
        Path(config['admin_token_file']).read_text().strip(),
        capacities,
        applications,
        limits=ServiceLimits(**config['limits']),
        allow_rebind=config.get('allow_rebind', []),
        application_policy=policy,
    )
    uvicorn.run(app, **listener, timeout_graceful_shutdown=15)


if __name__ == '__main__':
    main()
