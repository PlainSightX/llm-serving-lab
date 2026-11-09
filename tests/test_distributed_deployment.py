"""训练只能调度到声明的不同节点；rendezvous不向公网开放。"""

import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'deployment/n4.py'
    spec = importlib.util.spec_from_file_location('n4_deployment', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_two_node_rendezvous_and_hardware_roles_are_explicit():
    m = module()
    nodes = [{'node_name': 'a', 'instance_id': 'ia'}, {'node_name': 'b', 'instance_id': 'ib'}]
    items = m.training('alignment-a', 'ddp', 'correctness', nodes, '/owned', 'source-c')['items']
    service = next(x for x in items if x['kind'] == 'Service')
    jobs = [x for x in items if x['kind'] == 'Job']
    assert service['spec']['type'] == 'ClusterIP' and service['spec']['selector']['n4-rank'] == '0'
    assert len(jobs) == 2
    assert [
        x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in jobs
    ] == ['a', 'b']
    assert all(x['spec']['backoffLimit'] == 0 for x in jobs)


def test_duplicate_node_and_unapproved_mode_are_rejected():
    m = module()
    same = [{'node_name': 'a', 'instance_id': 'ia'}] * 2
    with pytest.raises(ValueError):
        m.training('bad', 'ddp', 'correctness', same, '/owned', 'source-c')
    with pytest.raises(ValueError):
        m.training('bad', 'single', 'correctness', same, '/owned', 'source-c')
