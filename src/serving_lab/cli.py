"""普通命令入口；所有写入都要求明确工作区与不可覆盖的身份。"""

import argparse
import json
import sys
from pathlib import Path

from .demo import run_demo
from .lab import Lab, LabError
from .reporting import write_report


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='经验巩固与受控模型更新工作台')
    parser.add_argument('--workspace', default='artifacts/workspace')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('status')
    p = sub.add_parser('demo')
    p.add_argument('--output', required=True)
    p = sub.add_parser('import')
    p.add_argument('path')
    p = sub.add_parser('task')
    p.add_argument('task_id')
    p = sub.add_parser('record')
    p.add_argument('attempt_id')
    p.add_argument('task_id')
    p.add_argument('model')
    p.add_argument('output')
    p.add_argument('--arm', default='frozen')
    p = sub.add_parser('feedback')
    p.add_argument('attempt_id')
    p = sub.add_parser('export')
    p.add_argument('--output', required=True)
    p.add_argument('--cutoff', type=int, required=True)
    p.add_argument('--arm', default='frozen')
    p = sub.add_parser('register')
    p.add_argument('model')
    p.add_argument('directory')
    p.add_argument('--fixture', action='store_true')
    p = sub.add_parser('register-base')
    p.add_argument('model')
    p.add_argument('directory')
    p = sub.add_parser('evaluate')
    p.add_argument('evaluation_id')
    p.add_argument('model')
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--arm', default='frozen')
    p = sub.add_parser('promote')
    p.add_argument('evaluation_id')
    p.add_argument('--expected-active')
    p.add_argument('--level', default='control_plane')
    p = sub.add_parser('rollback')
    p.add_argument('model')
    p.add_argument('--expected-active', required=True)
    p.add_argument('--level', default='control_plane')
    p = sub.add_parser('report')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            result = run_demo(args.output)
        else:
            with Lab(args.workspace) as lab:
                match args.command:
                    case 'init' | 'status':
                        result = lab.summary()
                    case 'import':
                        rows = [
                            json.loads(line)
                            for line in Path(args.path).read_text(encoding='utf-8').splitlines()
                            if line.strip()
                        ]
                        lab.import_tasks(rows)
                        result = {'imported': len(rows)}
                    case 'task':
                        result = lab.public_task(args.task_id)
                    case 'record':
                        lab.record(
                            args.attempt_id,
                            args.task_id,
                            args.model,
                            read_json(args.output),
                            arm=args.arm,
                            actor='manual',
                        )
                        result = {
                            'recorded': args.attempt_id,
                            'evidence': 'manual_unverified_execution',
                        }
                    case 'feedback':
                        result = lab.feedback(args.attempt_id)
                    case 'export':
                        result = lab.export_training(args.output, cutoff=args.cutoff, arm=args.arm)
                    case 'register':
                        result = lab.register(args.model, args.directory, fixture=args.fixture)
                    case 'register-base':
                        result = lab.register_base(args.model, args.directory)
                    case 'evaluate':
                        result = lab.evaluate(
                            args.evaluation_id, args.model, args.tasks, arm=args.arm
                        )
                    case 'promote':
                        result = {
                            'active': lab.promote(
                                args.evaluation_id,
                                expected_active=args.expected_active,
                                level=args.level,
                            ),
                            'scope': 'control_pointer_only',
                        }
                    case 'rollback':
                        result = {
                            'active': lab.rollback(
                                args.model, expected_active=args.expected_active, level=args.level
                            ),
                            'scope': 'control_pointer_only',
                        }
                    case 'report':
                        result = {'report': str(write_report(lab.summary(), args.output))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LabError, OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
