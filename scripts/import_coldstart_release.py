"""离线重核原始评价，再导出发布可消费的质量包；不运行模型或更改旧成绩。"""

import argparse
import json
from pathlib import Path

from serving_lab.application_release import ApplicationPolicy, BUNDLE_NAME, policy_document
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash, inspect_base
from serving_lab.parameters import parameter_identity
from q2_coldstart_audit import audit
from q2_teacher import read

REPO = Path(__file__).resolve().parents[1]


def import_evidence(baseline_run, trained_run, data, output):
    baseline, trained, data, out = map(
        lambda p: Path(p).resolve(), (baseline_run, trained_run, data, output)
    )
    out.mkdir(parents=True, exist_ok=False)
    certificates, checked = {}, {}
    sources = policy_document()['actor_sources']
    expected_controls = None
    for condition, root in [('base', baseline), ('trained', trained)]:
        result = audit(root, data, out / (condition + '-raw-audit.json'))
        initial, profile, binding = (
            read(root / 'input.json'),
            read(root / 'profile.json'),
            read(root / 'worker/binding.json'),
        )
        controls = initial['controls']
        if expected_controls is None:
            expected_controls = controls
        if controls != expected_controls:
            raise ValueError('两种参数没有使用同一开发合同')
        for name, h in sources.items():
            if initial['source_hashes']['src/serving_lab/' + name] != h:
                raise ValueError('actor源码变化；不能自动继承旧应用资格')
        model = Path(binding['model_path'])
        artifact = inspect_base(model)
        if digest(artifact['files']) != binding['artifact_digest']:
            raise ValueError('评价后的参数工件已变化')
        parameters = parameter_identity(model)
        if inspect_base(model) != artifact:
            raise ValueError('导入期间工件变化')
        metric = result['conditions'][condition]
        tasks = {t['task_id']: t for t in read(data / 'manifest.json')['tasks']}
        cases = [
            {
                'task_id': r['task_id'],
                'family': tasks[r['task_id']]['family'],
                'success': r['result']['success'],
            }
            for r in metric['tasks']
        ]
        certificates[binding['artifact_digest']] = {
            'artifact_digest': binding['artifact_digest'],
            'parameter_digest': parameters['digest'],
            'profile_digest': profile['digest'],
            'profile': profile['document'],
            'case_results': cases,
            'source': {
                'run': str(root),
                'condition': condition,
                'raw_audit_sha256': file_hash(out / (condition + '-raw-audit.json')),
                'input_sha256': file_hash(root / 'input.json'),
                'summary_sha256': file_hash(root / 'summary.json'),
                'model_manifest': artifact,
                'parameter_tensor_count': parameters['tensor_count'],
            },
        }
        checked[condition] = {
            'run': str(root),
            'successes': metric['successes'],
            'cases': metric['episodes'],
            'profile_digest': profile['digest'],
            'artifact_digest': binding['artifact_digest'],
        }
    interface_cases = []
    for i in range(1, 5):
        raw = read(trained / 'canaries' / str(i) / 'reservation.json')
        interface_cases.append(
            {
                'id': str(i),
                'messages': raw['wire']['messages'],
                'choices': raw['wire']['structured_outputs']['choice'],
            }
        )
    bundle = {
        'schema': 1,
        'name': BUNDLE_NAME,
        'policy': policy_document(),
        'certificates': certificates,
        'interface_panel': {'name': 'public_command_interface_v1', 'cases': interface_cases},
        'source_verification': {
            'producer': 'serving_lab.coldstart_release_import.v1',
            'evidence_level': 'real_local_model_evaluation',
            'data_manifest_sha256': file_hash(data / 'manifest.json'),
            'control_digest': digest(expected_controls),
            'checked': checked,
            'importer_sha256': file_hash(Path(__file__)),
            'new_model_calls': 0,
            'trust_boundary': 'operator pins imported bundle checksum; no signature or protection against malicious local administrator',
        },
    }
    bundle['bundle_digest'] = digest(bundle)
    policy = ApplicationPolicy(bundle)
    write_once(out / 'application-evidence.json', bundle)
    receipt = {
        'bundle': str(out / 'application-evidence.json'),
        'sha256': file_hash(out / 'application-evidence.json'),
        'bundle_digest': bundle['bundle_digest'],
        'decisions': {k: policy.summary(v) for k, v in certificates.items()},
        'new_model_calls': 0,
        'source_scope': 'frozen cold-start development panel; no production or general-chat claim',
    }
    write_once(out / 'import-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--trained', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(
        json.dumps(
            import_evidence(a.baseline, a.trained, a.data, a.output), ensure_ascii=False, indent=2
        )
    )
