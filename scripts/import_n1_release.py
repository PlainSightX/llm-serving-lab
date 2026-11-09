"""将独立复核的C1原应用评价追加到既有C0资格包；不改变质量门。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from serving_lab.application_release import ApplicationPolicy, policy_document
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_application_contract(parent, initial, expected_mode='application'):
    if (
        initial['mode'] != expected_mode
        or initial['data_sha256'] != parent['source_verification']['data_manifest_sha256']
        or initial.get('environment_contract_recorded') is not True
    ):
        raise ValueError('新增评价没有使用原应用的同一24题数据合同')
    for name, expected in parent['policy']['actor_sources'].items():
        if initial['source_hashes']['src/serving_lab/' + name] != expected:
            raise ValueError('两版本没有使用同一actor')


def import_evidence(
    parent_bundle, evaluation, releases, training, control_evaluation, control_releases, output
):
    parent_bundle, evaluation, releases, out = map(
        lambda p: Path(p).resolve(), (parent_bundle, evaluation, releases, output)
    )
    parent = read(parent_bundle)
    old_policy = ApplicationPolicy(parent)
    if parent['policy'] != policy_document():
        raise ValueError('原应用actor合同已变')
    initial = read(evaluation / 'input.json')
    verify_application_contract(parent, initial)
    proof = initial.get('candidate_training')
    if not proof or Path(proof['training']).resolve() != Path(training).resolve():
        raise ValueError('C1评价未绑定指定的已完成训练')
    binding = initial['worker_manifest']
    artifact = binding['artifact_digest']
    if artifact in parent['certificates'] or binding['parameter_digest'] in {
        c['parameter_digest'] for c in parent['certificates'].values()
    }:
        raise ValueError('必须是不同的真实参数版本')
    contract = lambda p: {k: v for k, v in p.items() if k != 'artifact_digest'}
    accepted = [c for c in parent['certificates'].values() if old_policy.summary(c)['passed']]
    if not accepted or not all(
        contract(c['profile']) == contract(binding['profile']) for c in accepted
    ):
        raise ValueError('新旧参数的公共profile合同不同，不能继承C0资格')
    out.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).with_name('n1_audit.py')),
        '--run',
        str(evaluation),
        '--releases',
        str(releases),
        '--output',
        str(out / 'candidate-raw-audit.json'),
    ]
    result = subprocess.run(command, text=True, encoding='utf-8', capture_output=True)
    (out / 'audit.stdout').write_text(result.stdout, encoding='utf-8')
    (out / 'audit.stderr').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('新参数原始评价复核失败')
    audit = read(out / 'candidate-raw-audit.json')
    if (
        not audit['raw_verified']
        or not audit['complete_run']
        or audit['mode'] != 'application'
        or audit['episodes'] != 24
    ):
        raise ValueError('新参数评价不完整')
    if (
        audit['artifact_digest'] != artifact
        or audit['parameter_digest'] != binding['parameter_digest']
    ):
        raise ValueError('审计参数身份不符')
    if audit.get('candidate_training') != proof or not proof['complete']:
        raise ValueError('独立审计未确认C1训练')
    cases = [
        {'task_id': r['task_id'], 'family': r['family'], 'success': r['success']}
        for r in audit['case_results']
    ]
    certificate = {
        'artifact_digest': artifact,
        'parameter_digest': binding['parameter_digest'],
        'profile_digest': binding['profile_digest'],
        'profile': binding['profile'],
        'case_results': cases,
        'source': {
            'run': str(evaluation),
            'condition': 'n1_old_application',
            'execution_location': 'cloud GPU; client and raw audit local',
            'raw_audit_sha256': file_hash(out / 'candidate-raw-audit.json'),
            'input_sha256': file_hash(evaluation / 'input.json'),
            'native_releases_sha256': file_hash(releases),
            'candidate_training': proof,
            'model_manifest': initial['reference_model'],
        },
    }
    control_evaluation, control_releases = (
        Path(control_evaluation).resolve(),
        Path(control_releases).resolve(),
    )
    control_input = read(control_evaluation / 'input.json')
    verify_application_contract(parent, control_input, 'application_control')
    control_binding = control_input['worker_manifest']
    control_artifact = control_binding['artifact_digest']
    if control_artifact != '1ee83b8562404b56bdb208a675bb5d4e4c83b827156df8f6b1e73d30842989de':
        raise ValueError('部署对照不是C0')
    if (
        control_artifact not in parent['certificates']
        or contract(control_binding['profile']) != contract(binding['profile'])
        or control_binding['parameter_digest']
        != parent['certificates'][control_artifact]['parameter_digest']
    ):
        raise ValueError('当前C0与原资格、C1公共profile不符')
    command = [
        sys.executable,
        str(Path(__file__).with_name('n1_audit.py')),
        '--run',
        str(control_evaluation),
        '--releases',
        str(control_releases),
        '--output',
        str(out / 'control-raw-audit.json'),
    ]
    result = subprocess.run(command, text=True, encoding='utf-8', capture_output=True)
    (out / 'control-audit.stdout').write_text(result.stdout, encoding='utf-8')
    (out / 'control-audit.stderr').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('当前部署C0原件复核失败')
    control = read(out / 'control-raw-audit.json')
    if (
        not control['raw_verified']
        or not control['complete_run']
        or control['mode'] != 'application_control'
        or control['episodes'] != 24
    ):
        raise ValueError('部署对照不完整')
    if (
        control['artifact_digest'] != control_artifact
        or control['parameter_digest'] != control_binding['parameter_digest']
    ):
        raise ValueError('当前C0审计参数不符')
    control_cert = {
        'artifact_digest': control_artifact,
        'parameter_digest': control_binding['parameter_digest'],
        'profile_digest': control_binding['profile_digest'],
        'profile': control_binding['profile'],
        'case_results': [
            {'task_id': r['task_id'], 'family': r['family'], 'success': r['success']}
            for r in control['case_results']
        ],
        'source': {
            'run': str(control_evaluation),
            'condition': 'current_cloud_c0_application',
            'raw_audit_sha256': file_hash(out / 'control-raw-audit.json'),
            'input_sha256': file_hash(control_evaluation / 'input.json'),
            'native_releases_sha256': file_hash(control_releases),
            'previous_certificate_digest': digest(parent['certificates'][control_artifact]),
        },
    }
    bundle = {
        **parent,
        'certificates': {
            **parent['certificates'],
            artifact: certificate,
            control_artifact: control_cert,
        },
        'source_verification': {
            'producer': 'serving_lab.application_release_import.v2',
            'evidence_level': 'real_local_and_cloud_model_evaluation',
            'parent_bundle_sha256': file_hash(parent_bundle),
            'parent_bundle_digest': parent['bundle_digest'],
            'parent_verification': parent['source_verification'],
            'added_audits': {
                artifact: file_hash(out / 'candidate-raw-audit.json'),
                control_artifact: file_hash(out / 'control-raw-audit.json'),
            },
            'data_manifest_sha256': initial['data_sha256'],
            'importer_sha256': file_hash(__file__),
            'new_model_calls': 0,
        },
    }
    bundle.pop('bundle_digest')
    bundle['bundle_digest'] = digest(bundle)
    policy = ApplicationPolicy(bundle)
    write_once(out / 'application-evidence.json', bundle)
    receipt = {
        'candidate': policy.summary(certificate),
        'current_control': policy.summary(control_cert),
        'bundle_sha256': file_hash(out / 'application-evidence.json'),
        'bundle_digest': bundle['bundle_digest'],
        'new_model_calls': 0,
        'quality_threshold_unchanged': True,
    }
    write_once(out / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--releases', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--control-evaluation', required=True)
    p.add_argument('--control-releases', required=True)
    a = p.parse_args()
    print(
        json.dumps(
            import_evidence(
                a.parent,
                a.evaluation,
                a.releases,
                a.training,
                a.control_evaluation,
                a.control_releases,
                a.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
