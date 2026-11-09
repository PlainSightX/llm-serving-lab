"""容器内私有环境桥；stdout仅输出公开视图，完整环境记录留在输出卷。"""

import contextlib
import hashlib
import json
import sys
from pathlib import Path

from serving_lab.public_tasks import PublicEpisode
from serving_lab.task_protocol import open_environment, canonical, validate_manifest


def main():
    request = json.loads(sys.stdin.readline())
    data = Path('/data')
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('name') == 'q2-public-teacher-coldstart-v1':
        from serving_lab.teacher_data import validate_teacher_manifest

        tasks = validate_teacher_manifest(manifest)
    elif manifest.get('name') == 'q2-public-workflows-v1':
        from serving_lab.workflow import validate_workflow_manifest

        tasks = validate_workflow_manifest(manifest)
    else:
        tasks = validate_manifest(manifest)
    task = tasks[request['task_id']]
    if task['role'] in {'protected', 'future'}:
        raise ValueError('保护或未来评价任务禁止在准备入口运行')
    with contextlib.redirect_stdout(sys.stderr):
        env = open_environment(task, data, reference=False)
    if manifest.get('name') == 'q2-public-workflows-v1':
        from serving_lab.workflow import WorkflowEpisode

        episode = WorkflowEpisode(
            env,
            request['max_steps'],
            task['workflow'],
            stagnation_limit=request.get('stagnation_limit'),
        )
    else:
        episode = PublicEpisode(
            env, request['max_steps'], stagnation_limit=request.get('stagnation_limit')
        )

    def send():
        value = {'public': episode.visible, 'terminal': episode.terminal}
        if 'workflow' in task:
            value['workflow'] = task['workflow']
        if episode.terminal:
            value['result'] = episode.result()
        print(canonical(value), flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            episode.reset()
        send()
        for line in sys.stdin:
            action = json.loads(line)['action']
            with contextlib.redirect_stdout(sys.stderr):
                episode.step(action)
            send()
            if episode.terminal:
                break
        record = {
            'task_id': task['task_id'],
            'initial': episode.initial,
            'trace': episode.trace,
            'result': episode.result() if episode.terminal else None,
            'reference_enabled': False,
        }
        # 只增加原件身份，不向actor暴露环境实现或隐藏状态。
        from serving_lab import public_tasks, task_protocol

        record['execution_contract'] = {
            'max_steps': request['max_steps'],
            'stagnation_limit': request.get('stagnation_limit'),
            'sources': {
                'scripts/q2_environment.py': hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                'src/serving_lab/public_tasks.py': hashlib.sha256(
                    Path(public_tasks.__file__).read_bytes()
                ).hexdigest(),
                'src/serving_lab/task_protocol.py': hashlib.sha256(
                    Path(task_protocol.__file__).read_bytes()
                ).hexdigest(),
            },
        }
        if 'workflow' in task:
            from serving_lab import workflow

            record['execution_contract']['sources']['src/serving_lab/workflow.py'] = hashlib.sha256(
                Path(workflow.__file__).read_bytes()
            ).hexdigest()
        if 'workflow' in task:
            record['workflow'] = task['workflow']
        with Path('/output/environment.json').open('x') as stream:
            stream.write(canonical(record) + '\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
