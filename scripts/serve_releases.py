"""维护者启动多版本入口；绑定和密钥只从本地受控配置读取。"""

import argparse
import json
from pathlib import Path

import uvicorn

from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.online import create_release_app, ServiceLimits
from serving_lab.parameters import bound_parameter_identity
from serving_lab.profile import TRANSPORT_SCOPE, application_history, check_runtime_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    panel = json.loads(Path(config['panel']).read_text(encoding='utf-8'))
    scoped = panel.get('qualification_scope') == TRANSPORT_SCOPE
    if scoped:
        check_runtime_contract(config['runtime'], TRANSPORT_SCOPE)
    engines = {}
    parameters = {}
    applications = {}
    for path in config['bindings']:
        binding = EngineBinding.load(path)
        if binding.version_id in engines:
            raise ValueError('重复实例版本')
        parameters[binding.version_id] = bound_parameter_identity(binding)
        engines[binding.version_id] = HttpEngine(binding, profile=scoped)
        if scoped:
            applications[binding.version_id] = application_history(
                config['application_receipts'][binding.version_id], binding
            )
    secret = Path(config['admin_token_file']).read_text().strip()
    app = create_release_app(
        engines,
        config['runtime'],
        parameters,
        panel,
        secret,
        limits=ServiceLimits(**config['limits']),
        allow_rebind=config.get('allow_rebind', []),
        application_results=applications if scoped else None,
    )
    if config.get('runtime_receipt'):
        from serving_lab.execution import write_once

        db = app.state.service.store.db
        settings = {
            'journal_mode': db.execute('PRAGMA journal_mode').fetchone()[0],
            'synchronous': db.execute('PRAGMA synchronous').fetchone()[0],
            'database_list': [list(r) for r in db.execute('PRAGMA database_list')],
            'runtime': config['runtime'],
        }
        if settings['journal_mode'] != 'wal' or settings['synchronous'] != 2:
            raise ValueError('测量要求实际WAL/FULL')
        write_once(config['runtime_receipt'], settings)
    uvicorn.run(app, host='127.0.0.1', port=config['port'], timeout_graceful_shutdown=15)


if __name__ == '__main__':
    main()
