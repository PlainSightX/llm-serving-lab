"""在模型所在机器启动受控worker桥；本包默认仅本机监听。"""

import argparse
from pathlib import Path
import uvicorn
from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.parameters import bound_parameter_identity
from serving_lab.worker_protocol import worker_manifest, create_worker_app
from serving_lab.execution import write_once
from serving_lab.tls import listener_options


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True)
    p.add_argument('--token-file', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--manifest-output', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--tls-ca')
    p.add_argument('--tls-cert')
    p.add_argument('--tls-key')
    p.add_argument(
        '--profile-policy',
        choices=['greedy_non_thinking_v1', 'q2_public_choice_v1', 'n2_length_controlled_v1'],
        default='greedy_non_thinking_v1',
    )
    a = p.parse_args()
    listener = listener_options(a.host, a.port, ca=a.tls_ca, cert=a.tls_cert, key=a.tls_key)
    policy = (
        None
        if a.profile_policy == 'greedy_non_thinking_v1'
        else {'name': 'q2_public_choice_v1', 'seed': 20261103}
    )
    if a.profile_policy == 'n2_length_controlled_v1':
        policy = {'name': a.profile_policy, 'seed': 0}
    binding = EngineBinding.load(a.binding)
    parameters = bound_parameter_identity(binding)
    engine = HttpEngine(binding, profile=True, profile_policy=policy)
    token = Path(a.token_file).read_text().strip()
    manifest = worker_manifest(engine, parameters['digest'])
    write_once(a.manifest_output, manifest)
    uvicorn.run(
        create_worker_app(engine, parameters['digest'], token),
        **listener,
        timeout_graceful_shutdown=15,
    )


if __name__ == '__main__':
    main()
