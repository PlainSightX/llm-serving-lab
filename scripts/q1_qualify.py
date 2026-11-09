"""在无网络CPU文本环境中运行Q1前检，不进行LLM推理或参数训练。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from serving_lab.task_protocol import (
    Episode,
    canonical,
    history_material,
    open_environment,
    build_context,
    validate_manifest,
)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')


def run_episode(task, data_root, origin, max_steps, actions=None, reference_method='handcoded'):
    env = None
    episode = None
    started = time.monotonic()
    try:
        env = open_environment(
            task,
            data_root,
            reference=origin == 'privileged_reference',
            reference_method=reference_method,
        )
        episode = Episode(
            task,
            env,
            max_steps=max_steps,
            origin=origin,
            reference_method=reference_method if origin == 'privileged_reference' else None,
        )
        episode.reset()
        index = 0
        while not episode.terminal:
            if origin == 'privileged_reference':
                action = episode.reference_action()
            elif actions is not None and index < len(actions):
                action = actions[index]
            else:
                action = 'look'
            episode.step(action)
            index += 1
        record = episode.record()
        return {'record': record, 'elapsed_seconds': time.monotonic() - started, 'error': None}
    except Exception as exc:
        return {
            'record': None,
            'elapsed_seconds': time.monotonic() - started,
            'error': {'type': type(exc).__name__, 'message': str(exc)},
            'partial_trace': episode.trace if episode else [],
            'task_id': task['task_id'],
            'origin': origin,
            'stage': 'episode_execution' if episode else 'environment_initialization',
        }
    finally:
        if episode:
            episode.close()
        elif env:
            env.close()


def isolated_episode(
    task, data_root, origin, max_steps, output, actions=None, reference_method='handcoded'
):
    """Fast Downward为每个环境复制本地库；回合进程退出后再回收确切临时目录。"""
    folder = output / task['task_id'] / (origin + '-process')
    folder.mkdir(parents=True, exist_ok=False)
    request = {
        'task': task,
        'data_root': str(data_root),
        'origin': origin,
        'max_steps': max_steps,
        'actions': actions,
        'reference_method': reference_method,
    }
    write_once(folder / 'input.json', request)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(output / 'source-code')
    with tempfile.TemporaryDirectory(prefix='q1-episode-', dir='/tmp') as temporary:
        env.update(TMPDIR=temporary, ALFWORLD_DATA=str(Path(temporary) / 'alfworld'))
        command = [
            sys.executable,
            str(output / 'source_q1_qualify.py'),
            '--episode-input',
            str(folder / 'input.json'),
            '--episode-result',
            str(folder / 'result.json'),
        ]
        try:
            result = subprocess.run(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
            )
            (folder / 'stdout.txt').write_bytes(result.stdout)
            (folder / 'stderr.txt').write_bytes(result.stderr)
            write_once(folder / 'exit.json', {'returncode': result.returncode})
        except subprocess.TimeoutExpired as exc:
            result_value = {
                'record': None,
                'partial_trace': [],
                'stage': 'episode_process',
                'error': {'type': 'TimeoutExpired', 'message': 'CPU episode exceeded 60 seconds'},
                'task_id': task['task_id'],
                'origin': origin,
            }
            write_once(folder / 'timeout.json', result_value)
            return result_value
    if result.returncode != 0 or not (folder / 'result.json').exists():
        return {
            'record': None,
            'partial_trace': [],
            'stage': 'environment_initialization',
            'error': {
                'type': 'ChildProcessError',
                'message': f'exit={result.returncode}; see process stderr',
            },
            'task_id': task['task_id'],
            'origin': origin,
        }
    return json.loads((folder / 'result.json').read_text(encoding='utf-8'))


def qualify(data_root, output, reference_method):
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'input.json').exists():
        raise ValueError('运行输出已存在，不覆盖首轮')
    manifest_path = data_root / 'manifest.json'
    expected = json.loads((data_root / 'manifest-anchor.json').read_text())['sha256']
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError('冻结任务清单已变化')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    (output / 'task-manifest.json').write_bytes(manifest_path.read_bytes())
    import serving_lab.task_protocol as module

    source = Path(module.__file__)
    write_once(
        output / 'input.json',
        {
            'manifest_sha256': expected,
            'task_protocol_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions': {
                name: importlib.metadata.version(name)
                for name in ('alfworld', 'textworld', 'numpy', 'pyyaml')
            },
            'max_steps': manifest['max_steps'],
            'reference_method': reference_method,
            'scope': 'reference and deterministic controls; no model learning claims',
        },
    )
    (output / 'source_task_protocol.py').write_bytes(source.read_bytes())
    (output / 'source_q1_qualify.py').write_bytes(Path(__file__).read_bytes())
    code = output / 'source-code/serving_lab'
    code.mkdir(parents=True)
    (code / '__init__.py').write_bytes(source.with_name('__init__.py').read_bytes())
    (code / 'task_protocol.py').write_bytes(source.read_bytes())
    references = {}
    rows = []
    for task in manifest['tasks']:
        if task['role'] == 'protected':
            continue
        folder = output / task['task_id']
        reference = isolated_episode(
            task,
            data_root,
            'privileged_reference',
            manifest['max_steps'],
            output,
            reference_method=reference_method,
        )
        write_once(folder / 'reference.json', reference)
        if reference.get('stage') == 'environment_initialization':
            write_once(output / 'environment-failure.json', reference)
            raise RuntimeError('任务环境初始化失败，停止整轮，不把运行错误算作任务失败')
        ref = reference['record']
        references[task['task_id']] = ref
        constant = isolated_episode(task, data_root, 'constant_look', manifest['max_steps'], output)
        write_once(folder / 'constant-look.json', constant)
        controls = [reference, constant]
        checks = {
            'reference_success': bool(ref and ref['result']['success']),
            'constant_success': bool(
                constant['record'] and constant['record']['result']['success']
            ),
            'constant_valid_rate': constant['record']['result']['format_valid_rate']
            if constant['record']
            else None,
            'initial_observation_equal': bool(
                ref and constant['record'] and ref['initial'] == constant['record']['initial']
            ),
        }
        if ref and ref['result']['success']:
            actions = [item['action'] for item in ref['trace']]
            changed = actions[:-1] + ['look']
            counter = isolated_episode(
                task, data_root, 'trajectory_counterfactual', len(changed), output, changed
            )
            controls.append(counter)
            write_once(
                folder / 'last-action-counterfactual.json',
                {'replaced_step': len(changed), 'original_final_action': actions[-1], **counter},
            )
            checks['counterfactual_success'] = bool(
                counter['record'] and counter['record']['result']['success']
            )
            checks['counterfactual_initial_equal'] = bool(
                counter['record'] and counter['record']['initial'] == ref['initial']
            )
            checks['counterfactual_prefix_equal'] = bool(
                counter['record'] and counter['record']['trace'][:-1] == ref['trace'][:-1]
            )
        if task['role'] == 'target':
            if ref:
                write_once(
                    folder / 'input-none.json',
                    build_context(ref['initial'], None, condition='none', cutoff=task['sequence']),
                )
            support = references.get(task['candidate_source_task_id'])
            if support and support['result']['success']:
                material = history_material(support, cutoff=task['sequence'])
                write_once(folder / 'related-reference-history.json', material)
                if ref:
                    write_once(
                        folder / 'input-oracle-related.json',
                        build_context(
                            ref['initial'],
                            material,
                            condition='oracle_related',
                            cutoff=task['sequence'],
                        ),
                    )
                alternative = next(
                    (
                        r
                        for r in references.values()
                        if r
                        and r['task']['role'] == 'support'
                        and r['task']['family'] != task['family']
                        and r['result']['success']
                    ),
                    None,
                )
                if alternative:
                    alternative_material = history_material(alternative, cutoff=task['sequence'])
                    write_once(
                        folder / 'different-family-reference-history.json', alternative_material
                    )
                    if ref:
                        write_once(
                            folder / 'input-different-family.json',
                            build_context(
                                ref['initial'],
                                alternative_material,
                                condition='different_family_reference',
                                cutoff=task['sequence'],
                            ),
                        )
                replay = isolated_episode(
                    task,
                    data_root,
                    'literal_replay',
                    manifest['max_steps'],
                    output,
                    [step['action'] for step in support['trace']],
                )
                controls.append(replay)
                write_once(folder / 'literal-replay.json', replay)
                checks['literal_replay_success'] = bool(
                    replay['record'] and replay['record']['result']['success']
                )
                checks['history_origin'] = 'privileged_reference'
                checks['history_characters'] = len(canonical(material))
                checks['history_token_budget_bound'] = False
        errors = [control['error'] for control in controls if control['error']]
        rows.append(
            {
                'task_id': task['task_id'],
                'role': task['role'],
                'family': task['family'],
                'reference_steps': ref['result']['steps'] if ref else None,
                'checks': checks,
                'errors': errors,
            }
        )
        print(canonical(rows[-1]), flush=True)
    valid = len(rows) == 12 and all(
        r['checks']['reference_success']
        and not r['checks']['constant_success']
        and r['checks']['constant_valid_rate'] == 1
        and r['checks']['initial_observation_equal']
        and r['checks'].get('counterfactual_initial_equal')
        and r['checks'].get('counterfactual_prefix_equal')
        and not r['checks'].get('counterfactual_success')
        and not r['errors']
        for r in rows
    )
    summary = {
        'environment_preflight_passed': valid,
        'tasks_executed': len(rows),
        'protected_tasks_executed': 0,
        'new_model_calls': 0,
        'training_runs': 0,
        'rows': rows,
        'scope': 'CPU task/evaluation preflight only; related history benefits and procedural transfer remain hypotheses',
        'reference_origin': f'privileged {reference_method}, not an equal-information baseline or model experience',
        'counterfactual_scope': 'same trajectory prefix, final action replaced; not a rule-change benchmark',
        'history_budget': 'character sizes are descriptive; Q2 must bind actual tokenizer and token-matched controls',
    }
    write_once(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output')
    parser.add_argument('--episode-input')
    parser.add_argument('--episode-result')
    parser.add_argument('--reference-method', choices=['handcoded', 'planner'], default='handcoded')
    args = parser.parse_args()
    if args.episode_input:
        request = json.loads(Path(args.episode_input).read_text(encoding='utf-8'))
        result = run_episode(
            request['task'],
            request['data_root'],
            request['origin'],
            request['max_steps'],
            request['actions'],
            request.get('reference_method', 'handcoded'),
        )
        write_once(args.episode_result, result)
        raise SystemExit(0)
    result = qualify(args.data, args.output, args.reference_method)
    print(canonical({k: v for k, v in result.items() if k != 'rows'}))
    raise SystemExit(0 if result['environment_preflight_passed'] else 2)
