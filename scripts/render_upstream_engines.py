"""校验固定chart中的双副本vLLM配置；不会安装或连接任何集群。"""

import argparse
import json
import subprocess
from pathlib import Path
import yaml

from serving_lab.paths import lab_root


def run(chart, output):
    repo = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    helm = str(lab_root() / 'bin/helm-v3.19.0')
    values = [
        repo / 'configs/upstream/router-contract-values.yaml',
        repo / 'configs/upstream/two-replica-values.yaml',
    ]
    command = [
        helm,
        'template',
        'lsl-upstream',
        str(Path(chart).resolve()),
        '--namespace',
        'lsl-inference',
    ]
    for p in values:
        command += ['-f', str(p)]
        (output / p.name).write_bytes(p.read_bytes())
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (output / 'rendered.yaml').write_text(result.stdout)
    (output / 'stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Helm engine render failed')
    docs = [v for v in yaml.safe_load_all(result.stdout) if v]
    deployments = [d for d in docs if d['kind'] == 'Deployment']
    engines = [
        d
        for d in deployments
        if any(
            'vllm/vllm-openai' in c['image'] for c in d['spec']['template']['spec']['containers']
        )
    ]
    assert len(deployments) == 2 and len(engines) == 1
    engine = engines[0]
    pod = engine['spec']['template']['spec']
    c = pod['containers'][0]
    args = c.get('command', []) + c.get('args', [])
    env = {e['name']: e.get('value') for e in c['env']}
    assert (
        engine['spec']['replicas'] == 2 and str(c['resources']['limits']['nvidia.com/gpu']) == '1'
    )
    assert engine['spec']['strategy']['rollingUpdate'] == {'maxSurge': 0, 'maxUnavailable': 1}
    assert (
        'v0.11.2@sha256:2c908d5a84ed251b6a17d179f42d06df1aff353007779ac5eecd8a0ea3fe9331'
        in c['image']
    )
    assert env.get('VLLM_USE_V1') != '0' and env['HF_HUB_OFFLINE'] == '1'
    assert '--no-enable-prefix-caching' in args and '--enable-prefix-caching' not in args
    assert (
        args[args.index('--generation-config') + 1] == 'vllm'
        and args[args.index('--served-model-name') + 1] == 'qwen3-06b-r2'
    )
    assert (
        pod['affinity']['podAntiAffinity']['requiredDuringSchedulingIgnoredDuringExecution'][0][
            'topologyKey'
        ]
        == 'kubernetes.io/hostname'
    )
    assert any(
        v.get('persistentVolumeClaim', {}).get('claimName') == 'qwen3-06b-models-rwx'
        for v in pod['volumes']
    )
    assert any(v['mountPath'] == '/data' for v in c['volumeMounts'])
    (output / 'receipt.json').write_text(
        json.dumps(
            {
                'verified_render': True,
                'command': command,
                'engine_replicas': 2,
                'gpu_per_replica': 1,
                'cross_hostname_required': True,
                'no_model_download_configured': True,
                'requires': 'two actual GPU nodes, nvidia runtime and preprovisioned RWX model claim',
                'scope': 'render only, no image pull / install / runtime compatibility claim',
            },
            indent=2,
        )
        + '\n'
    )
    (output / 'source_render.py').write_bytes(Path(__file__).read_bytes())
    print('two-replica pinned engine configuration rendered and checked; no deployment')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chart', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.chart, a.output)
