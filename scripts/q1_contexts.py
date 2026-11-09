"""从冻结episode派生诊断输入；加强消费资格不改写原观察/动作/评价。"""

import argparse
import hashlib
import json
from pathlib import Path

from serving_lab.task_protocol import (
    canonical,
    history_material,
    build_context,
    validate_manifest,
    validate_record,
)


def verify_execution_sources(episodes, execution):
    """核对运行记录绑定的源码，避免导出阶段接纳已漂移的执行文件。"""
    for name, key in [
        ('source_task_protocol.py', 'task_protocol_sha256'),
        ('source_q1_qualify.py', 'runner_sha256'),
        ('source-code/serving_lab/task_protocol.py', 'task_protocol_sha256'),
    ]:
        if hashlib.sha256((episodes / name).read_bytes()).hexdigest() != execution[key]:
            raise ValueError('执行源码快照不符')


def validated_records(episodes, manifest_path=None):
    """输入锚点、完整任务覆盖、原始子回合和派生视图共同核验。"""
    episodes = Path(episodes).resolve()
    execution = json.loads((episodes / 'input.json').read_text(encoding='utf-8'))
    manifest_path = Path(manifest_path) if manifest_path else episodes / 'task-manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != execution['manifest_sha256']:
        raise ValueError('任务清单不匹配运行时锚点')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_id = validate_manifest(manifest)
    summary = json.loads((episodes / 'summary.json').read_text(encoding='utf-8'))
    expected = {key for key, t in by_id.items() if t['role'] != 'protected'}
    observed = [r['task_id'] for r in summary['rows']]
    if (
        set(observed) != expected
        or len(observed) != len(expected)
        or summary['tasks_executed'] != len(expected)
    ):
        raise ValueError('运行任务覆盖不完整')
    verify_execution_sources(episodes, execution)
    for task in by_id.values():
        if task['role'] == 'protected' and (episodes / task['task_id']).exists():
            raise ValueError('保护任务存在执行输出')
    records = {}
    sources = {}
    for row in summary['rows']:
        task = by_id[row['task_id']]
        folder = episodes / task['task_id']
        checked = {}
        if row['role'] != task['role'] or row['family'] != task['family']:
            raise ValueError('摘要任务元数据与清单不符')
        policies = [
            ('privileged_reference', 'reference.json'),
            ('constant_look', 'constant-look.json'),
            ('trajectory_counterfactual', 'last-action-counterfactual.json'),
        ]
        if task['role'] == 'target':
            policies.append(('literal_replay', 'literal-replay.json'))
        for origin, name in policies:
            path = folder / name
            view = json.loads(path.read_text(encoding='utf-8'))
            raw_path = folder / (origin + '-process') / 'result.json'
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
            record = validate_record(view['record'], task)
            if (
                view['error'] is not None
                or raw['error'] is not None
                or record != raw['record']
                or record['origin'] != origin
            ):
                raise ValueError('原始子回合与派生记录不一致')
            exit_path = raw_path.parent / 'exit.json'
            if not exit_path.exists():
                raise ValueError('旧回合未记录退出码，不能自动升级当前完整核验')
            if json.loads(exit_path.read_text())['returncode'] != 0:
                raise ValueError('子回合异常退出')
            if (
                origin == 'privileged_reference'
                and record.get('reference_method') != execution['reference_method']
            ):
                raise ValueError('参考策略不符')
            if (
                origin != 'trajectory_counterfactual'
                and record['result']['max_steps'] != manifest['max_steps']
            ):
                raise ValueError('回合预算不符')
            checked[origin] = record
            for source in (path, raw_path):
                sources[str(source)] = {
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'record_hash': record['record_hash'],
                }
        ref = checked['privileged_reference']
        constant = checked['constant_look']
        counter = checked['trajectory_counterfactual']
        if (
            not ref['result']['success']
            or constant['result']['success']
            or constant['result']['format_valid_rate'] != 1
            or counter['result']['success']
            or constant['initial'] != ref['initial']
            or counter['initial'] != ref['initial']
            or counter['trace'][:-1] != ref['trace'][:-1]
            or counter['result']['steps'] != ref['result']['steps']
            or counter['trace'][-1]['action'] != 'look'
        ):
            raise ValueError('重新计算的前检条件未通过')
        computed = {
            'reference_success': True,
            'constant_success': False,
            'constant_valid_rate': 1.0,
            'initial_observation_equal': True,
            'counterfactual_success': False,
            'counterfactual_initial_equal': True,
            'counterfactual_prefix_equal': True,
        }
        if task['role'] == 'target':
            computed['literal_replay_success'] = checked['literal_replay']['result']['success']
        if row['reference_steps'] != ref['result']['steps'] or any(
            row['checks'].get(k) != v for k, v in computed.items()
        ):
            raise ValueError('摘要指标与原始回合不符')
        records[task['task_id']] = ref
    return manifest, summary, records, sources


def derive(episodes, output, manifest_path=None):
    episodes = Path(episodes).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('派生目录已存在，保留原版本')
    _, summary, records, sources = validated_records(episodes, manifest_path)
    output.mkdir(parents=True)
    count = 0
    for row in summary['rows']:
        if row['role'] != 'target':
            continue
        target = records[row['task_id']]
        task = target['task']
        relevant = records[task['candidate_source_task_id']]
        control = next(
            r
            for r in records.values()
            if r['task']['role'] == 'support' and r['task']['family'] != task['family']
        )
        folder = output / task['task_id']
        folder.mkdir()
        conditions = {
            'none': None,
            'oracle_related': history_material(
                relevant, cutoff=task['sequence'], purpose='diagnostic'
            ),
            'different_family_reference': history_material(
                control, cutoff=task['sequence'], purpose='diagnostic'
            ),
        }
        for name, history in conditions.items():
            context = build_context(
                target['initial'],
                history,
                condition=name,
                cutoff=task['sequence'],
                purpose='diagnostic',
            )
            (folder / (name + '.json')).write_text(canonical(context) + '\n', encoding='utf-8')
            count += 1
    import serving_lab.task_protocol as module

    manifest = {
        'schema': 3,
        'source_episode_directory': str(episodes),
        'source_files': sources,
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy_source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'contexts': count,
        'model_trial_ready': False,
        'history_role': 'oracle_diagnostic_only',
        'note': '原episode与原上下文保留；本版逐条核验完整冻结任务与原始子回合，未追加环境或模型调用。',
    }
    (output / 'source_q1_contexts.py').write_bytes(Path(__file__).read_bytes())
    (output / 'source_task_protocol.py').write_bytes(Path(module.__file__).read_bytes())
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest')
    args = parser.parse_args()
    result = derive(args.episodes, args.output, args.manifest)
    print(canonical({k: v for k, v in result.items() if k != 'source_files'}))
