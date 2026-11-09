"""只清理明确归属本次运行的容器，诊断失败单独留证。"""

import json
from pathlib import Path

from serving_lab.execution import write_once


def cleanup_owned(
    docker, containers, owner, network, output, *, stop_timeout=None, diagnostic_errors=None
):
    output = Path(output)
    report = {
        'containers': [],
        'network_removed': False,
        'errors': [],
        'diagnostic_errors': list(diagnostic_errors or []),
    }
    for role, cid in reversed(containers):
        row = {'role': role, 'id': cid, 'removed': False, 'state': None}
        try:
            original = json.loads(docker('inspect', cid).stdout)[0]
            if original['Config']['Labels'].get('serving-lab.run_id') != owner:
                raise ValueError('容器所有权不符')
            if stop_timeout is not None:
                docker('stop', '--time', str(stop_timeout), cid, check=False)
            current = json.loads(docker('inspect', cid).stdout)[0]
            row['state'] = current['State']
        except Exception as error:
            report['errors'].append(
                {
                    'role': role,
                    'phase': 'ownership_or_stop',
                    'type': type(error).__name__,
                    'message': str(error),
                }
            )
            report['containers'].append(row)
            continue
        try:
            logs = docker('logs', cid, check=False)
            (output / (role + '.log')).write_text(logs.stdout + logs.stderr, encoding='utf-8')
        except Exception as error:
            report['diagnostic_errors'].append(
                {'role': role, 'type': type(error).__name__, 'message': str(error)}
            )
        try:
            removed = docker('rm', '-f', '-v', cid, check=False)
            row['removed'] = removed.returncode == 0
            if not row['removed']:
                raise RuntimeError('Docker未确认容器移除')
        except Exception as error:
            report['errors'].append(
                {
                    'role': role,
                    'phase': 'remove',
                    'type': type(error).__name__,
                    'message': str(error),
                }
            )
        report['containers'].append(row)
    if network:
        try:
            info = json.loads(docker('network', 'inspect', network).stdout)[0]
            if info['Labels'].get('serving-lab.run_id') != owner:
                raise ValueError('网络所有权不符')
            docker('network', 'rm', network)
            report['network_removed'] = True
        except Exception as error:
            report['errors'].append(
                {'phase': 'network', 'type': type(error).__name__, 'message': str(error)}
            )
    write_once(output / 'cleanup.json', report)
    if report['errors'] or not all(row['removed'] for row in report['containers']):
        raise RuntimeError('自有资源清理未完成，详见cleanup.json')
    return report
