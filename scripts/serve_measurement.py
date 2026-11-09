"""N2受控测量入口；传输资格不被标成业务应用资格。"""

import argparse
import json
from pathlib import Path
import uvicorn
from serving_lab.worker_protocol import RemoteEngine
from serving_lab.replica_api import create_replica_app
from serving_lab.online import ServiceLimits
from serving_lab.tls import listener_options
from serving_lab.lab import file_hash


def main(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('scope') != 'n2_measurement_only_v1':
        raise ValueError('非N2测量合同')
    engines = {}
    parameters = {}
    capacities = {}
    applications = {}
    for item in config['replicas']:
        if file_hash(item['manifest']) != item['manifest_sha256']:
            raise ValueError('测量worker清单变化')
        manifest = json.loads(Path(item['manifest']).read_text(encoding='utf-8'))
        if (
            manifest['profile']['name'] != 'n2_length_controlled_v1'
            or manifest['artifact_digest'] != item['expected_artifact_digest']
        ):
            raise ValueError('测量profile或参数工件不符')
        engine = RemoteEngine(
            item['endpoint'],
            Path(item['token_file']).read_text().strip(),
            manifest,
            tls=item['tls'],
        )
        engines[item['id']] = engine
        parameters[engine.version_id] = {'digest': engine.parameter_digest}
        capacities[item['id']] = item['capacity']
        applications[engine.version_id] = {
            'scope': 'n2_measurement_only_v1',
            'application_qualified': False,
            'meaning': 'quality is independently checked by the frozen study; this endpoint proves transport and resource behavior',
        }
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
    )
    uvicorn.run(
        app,
        **listener_options(config['host'], config['port'], **config['tls']),
        timeout_graceful_shutdown=15,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    main(a.config)
