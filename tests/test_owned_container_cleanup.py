"""诊断异常不会阻止其他自有容器清理，所有权不符则保留该对象。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class Docker:
    def __init__(self):
        self.owners = {'a': 'run', 'b': 'run'}
        self.running = {'a': True, 'b': True}
        self.removed = []

    def __call__(self, *args, check=True):
        if args[0] == 'inspect':
            cid = args[1]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            'Config': {'Labels': {'serving-lab.run_id': self.owners[cid]}},
                            'State': {'Running': self.running[cid], 'ExitCode': 0},
                        }
                    ]
                ),
                stderr='',
            )
        if args[0] == 'stop':
            self.running[args[-1]] = False
        if args[0] == 'logs':
            raise UnicodeDecodeError('utf-8', b'private-test-key', 0, 1, 'invalid byte')
        if args[0] == 'rm':
            self.removed.append(args[-1])
        if args[:2] == ('network', 'inspect'):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{'Labels': {'serving-lab.run_id': 'run'}}]),
                stderr='',
            )
        return SimpleNamespace(returncode=0, stdout='', stderr='')


def test_log_decode_failure_does_not_interrupt_cleanup(tmp_path):
    from owned_containers import cleanup_owned

    docker = Docker()
    result = cleanup_owned(
        docker, [('worker', 'a'), ('gateway', 'b')], 'run', 'network', tmp_path, stop_timeout=1
    )
    assert set(docker.removed) == {'a', 'b'} and result['network_removed']
    assert len(result['diagnostic_errors']) == 2 and not result['errors']
    assert 'private-test-key' not in json.dumps(result)


def test_ownership_error_keeps_foreign_container_and_cleans_other_owned_resource(tmp_path):
    from owned_containers import cleanup_owned

    docker = Docker()
    docker.owners['b'] = 'foreign'
    with pytest.raises(RuntimeError, match='清理'):
        cleanup_owned(
            docker, [('worker', 'a'), ('foreign', 'b')], 'run', None, tmp_path, stop_timeout=1
        )
    assert docker.removed == ['a'] and docker.running['b']
