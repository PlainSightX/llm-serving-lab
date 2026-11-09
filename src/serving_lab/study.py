"""业务协议试点入口：冻结两组开发流，保护流不执行，不把试点晋级成研究。"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

from .business import audit_suite, build_suite
from .execution import DEFAULT_GENERATION, verify_execution, write_once
from .lab import Lab, LabError, canonical, check_output, digest, file_hash


def prepare(directory, model_path):
    directory = Path(directory).resolve()
    if directory.exists():
        raise LabError('试点目录已存在；保留旧输入与失败')
    tasks = build_suite()
    audit = audit_suite(tasks)
    directory.mkdir(parents=True)
    files = {}
    for name, rows in [
        ('suite.jsonl', tasks),
        ('pilot.jsonl', [t for t in tasks if t['split'] != 'holdout']),
        ('protected.jsonl', [t for t in tasks if t['split'] == 'holdout']),
    ]:
        path = directory / name
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(''.join(canonical(t) + '\n' for t in rows))
        files[name] = file_hash(path)
    pilot = [t for t in tasks if t['split'] != 'holdout']
    identities = []
    arms = {'frozen': {'memory_policy': 'none'}, 'memory': {'memory_policy': 'episodic_v1'}}
    spec = {**DEFAULT_GENERATION, 'max_new_tokens': 256, 'max_input_tokens': 4096}
    for arm in arms:
        with Lab(directory / arm) as lab:
            lab.import_tasks(pilot)
            model = lab.register_base('base', model_path)
            identities.append(model['parameter_identity'])
            with lab.db:
                for key, value in {
                    'arm': arm,
                    'memory_policy': arms[arm]['memory_policy'],
                    'generation': canonical(spec),
                    'study_model': 'base',
                }.items():
                    lab.db.execute('INSERT INTO metadata VALUES(?,?)', (key, value))
    if len(set(identities)) != 1:
        raise LabError('两个臂的实际基座不同')
    config = {
        'schema': 1,
        'stage': 'business_protocol_pilot',
        'dataset': audit,
        'files': files,
        'arms': arms,
        'parameter_identity': identities[0],
        'model_id': 'base',
        'max_new_tokens': 256,
        'max_input_tokens': 4096,
        'device': 'cuda',
        'scheduled_per_arm': len(pilot),
        'created': time.time(),
        'trajectory': 'natural_closed_loop',
        'protected_executed': False,
        'limits': '两组真实业务协议资格；同一合成计划语言，尚非强记忆/持续学习效果比较。三组权重更新臂尚待业务与数据资格。',
    }
    write_once(directory / 'study.json', config)
    write_once(
        directory / 'study-anchor.json', {'study_sha256': file_hash(directory / 'study.json')}
    )
    return config


def verify_study(directory):
    directory = Path(directory)
    anchor = json.loads((directory / 'study-anchor.json').read_text(encoding='utf-8'))
    if file_hash(directory / 'study.json') != anchor['study_sha256']:
        raise LabError('冻结试点配置已修改')
    config = json.loads((directory / 'study.json').read_text(encoding='utf-8'))
    for name, expected in config['files'].items():
        if file_hash(directory / name) != expected:
            raise LabError('冻结试点数据已修改')
    tasks = [
        json.loads(line)
        for line in (directory / 'suite.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    if audit_suite(tasks) != config['dataset']:
        raise LabError('数据审计不一致')
    pilot = [t for t in tasks if t['split'] != 'holdout']
    if [
        json.loads(line)
        for line in (directory / 'pilot.jsonl').read_text(encoding='utf-8').splitlines()
    ] != pilot:
        raise LabError('开发执行流与总清单不一致')
    for arm in config['arms']:
        with Lab(directory / arm) as lab:
            actual = [
                lab.task(r['id']) for r in lab.db.execute('SELECT id FROM tasks ORDER BY seq')
            ]
            if actual != pilot:
                raise LabError('工作区任务与冻结清单不符')
            if lab.model(config['model_id'])['parameter_identity'] != config['parameter_identity']:
                raise LabError('实际模型与试点不符')
            spec = {
                **DEFAULT_GENERATION,
                'device': config['device'],
                'max_new_tokens': config['max_new_tokens'],
                'max_input_tokens': config['max_input_tokens'],
            }
            required = {
                'arm': arm,
                'memory_policy': config['arms'][arm]['memory_policy'],
                'generation': canonical(spec),
                'study_model': config['model_id'],
            }
            metadata = dict(lab.db.execute('SELECT key,value FROM metadata'))
            if any(metadata.get(k) != v for k, v in required.items()):
                raise LabError('工作区未绑定预定比较臂合同')
            for row in lab.db.execute(
                'SELECT a.*,e.request_hash,e.directory FROM attempts a LEFT JOIN executions e ON e.attempt_id=a.id'
            ):
                if (
                    row['actor'] != 'local_hf'
                    or row['model'] != config['model_id']
                    or row['arm'] != arm
                    or not row['directory']
                ):
                    raise LabError('试点中存在非预定模型/来源/比较臂尝试')
                path = Path(row['directory']) / 'request.json'
                if file_hash(path) != row['request_hash']:
                    raise LabError('试点请求已修改')
                req = json.loads(path.read_text(encoding='utf-8'))
                if (
                    req['generation'] != spec
                    or req['arm'] != arm
                    or req['model_id'] != config['model_id']
                    or req['parameter_identity'] != config['parameter_identity']
                    or req['context']['policy'] != required['memory_policy']
                ):
                    raise LabError('实际请求与冻结试点不一致')
    return config


def run(directory, arm):
    directory = Path(directory)
    config = verify_study(directory)
    if arm not in config['arms']:
        raise LabError('未知比较臂')
    from .inference import run_tasks

    started = time.perf_counter()
    result = run_tasks(
        directory / arm,
        config['model_id'],
        'run',
        device=config['device'],
        max_new_tokens=config['max_new_tokens'],
        max_input_tokens=config['max_input_tokens'],
        arm=arm,
        memory_policy=config['arms'][arm]['memory_policy'],
        resume=True,
    )
    result['runner_elapsed_seconds'] = time.perf_counter() - started
    write_once(directory / arm / f'run-summary-{time.time_ns()}.json', result)
    return result


def report(directory):
    directory = Path(directory)
    config = verify_study(directory)
    summary = {
        'stage': config['stage'],
        'dataset': config['dataset'],
        'limits': config['limits'],
        'arms': {},
        'all_scheduled_completed': True,
        'protected_executed': False,
    }
    for arm in config['arms']:
        value = {
            'scheduled': config['scheduled_per_arm'],
            'completed_model_calls': 0,
            'reserved_attempts': 0,
            'passed': 0,
            'failed_or_missing': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'generation_seconds': 0.0,
            'with_memory': 0,
            'by_split': {},
            'by_version': {},
            'errors': {},
            'tasks': [],
        }
        with Lab(directory / arm) as lab:
            cache = {}
            for taskrow in lab.db.execute('SELECT id FROM tasks ORDER BY seq'):
                task = lab.task(taskrow['id'])
                row = lab.db.execute(
                    'SELECT * FROM attempts WHERE task_id=? ORDER BY created,id LIMIT 1',
                    (task['task_id'],),
                ).fetchone()
                passed = False
                errors = ['missing']
                state = 'missing'
                mem = []
                if row and row['actor'] == 'local_hf':
                    value['reserved_attempts'] += 1
                    verified = verify_execution(lab, row['id'], _cache=cache)
                    state = verified['state']
                    errors = [state]
                    if verified['complete']:
                        value['completed_model_calls'] += 1
                        verdict = check_output(task, json.loads(row['output']))
                        passed = (
                            verdict['passed']
                            and verified['output_complete']
                            and verified['json_valid']
                        )
                        errors = verdict.get('errors', [])[:]
                        if not verified['output_complete']:
                            errors.append('truncated')
                        if not verified['json_valid']:
                            errors.append('invalid_json')
                        path = directory / arm / 'executions' / row['id']
                        response = json.loads((path / 'response.json').read_text(encoding='utf-8'))
                        req = json.loads((path / 'request.json').read_text(encoding='utf-8'))
                        runtime = response['runtime']
                        value['prompt_tokens'] += runtime['prompt_tokens']
                        value['completion_tokens'] += runtime['completion_tokens']
                        value['generation_seconds'] += response['elapsed_seconds']
                        mem = [e['attempt_id'] for e in req['context']['entries']]
                        if mem:
                            value['with_memory'] += 1
                value['passed'] += int(passed)
                for grouping, key in [
                    ('by_split', task['split']),
                    ('by_version', task['rule_version']),
                ]:
                    group = value[grouping].setdefault(key, {'passed': 0, 'total': 0})
                    group['passed'] += int(passed)
                    group['total'] += 1
                for error in errors:
                    value['errors'][error] = value['errors'].get(error, 0) + 1
                value['tasks'].append(
                    {
                        'task_id': task['task_id'],
                        'split': task['split'],
                        'rule_version': task['rule_version'],
                        'state': state,
                        'passed': passed,
                        'errors': errors,
                        'memory_sources': mem,
                    }
                )
        value['failed_or_missing'] = value['scheduled'] - value['passed']
        value['incomplete_attempts'] = value['reserved_attempts'] - value['completed_model_calls']
        value['cost_scope'] = (
            'generation_seconds只含已完成解码，不含加载/验证/检索/中断消耗；完整入口时间见各run-summary'
        )
        if value['completed_model_calls'] != value['scheduled']:
            summary['all_scheduled_completed'] = False
        summary['arms'][arm] = value
    # 报告是可重建视图；原始输入、首尝和回执保持不可覆盖。
    (directory / 'report.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    rows = ''.join(
        f'<tr><td>{html.escape(arm)}</td><td>{v["passed"]}/{v["scheduled"]}</td><td>{v["completed_model_calls"]}</td><td>{v["with_memory"]}</td><td>{v["prompt_tokens"]}</td><td>{v["completion_tokens"]}</td></tr>'
        for arm, v in summary['arms'].items()
    )
    page = '<!doctype html><meta charset="utf-8"><title>P2 业务试点</title><style>body{max-width:1050px;margin:40px auto;font-family:system-ui;line-height:1.6}td,th{padding:8px 16px;border:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
    page += '<h1>风电实验比较：业务协议试点</h1><p>' + html.escape(summary['limits']) + '</p>'
    page += (
        '<table><tr><th>组别</th><th>合格/计划任务</th><th>完成模型调用</th><th>使用历史的任务</th><th>输入token</th><th>输出token</th></tr>'
        + rows
        + '</table>'
    )
    page += (
        '<p>当前规则对两组相同。独立工作区自然闭环；不是相同历史下的因果消融。保护面板没有执行。原始记录按任务保留。</p><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + '</pre>'
    )
    (directory / 'report.html').write_text(page, encoding='utf-8')
    return summary


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='P2业务协议与两组真实开发流')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p = commands.add_parser('run')
    p.add_argument('--directory', required=True)
    p.add_argument('--arm', choices=['frozen', 'memory'], required=True)
    p = commands.add_parser('report')
    p.add_argument('--directory', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.output, args.model)
        elif args.command == 'run':
            result = run(args.directory, args.arm)
        else:
            result = report(args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LabError, OSError, RuntimeError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
