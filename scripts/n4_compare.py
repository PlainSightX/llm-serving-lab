"""在同一冻结合同下比较完整参数、末步梯度和逐步loss，不以摘要passed替代。"""

import argparse
import json
from pathlib import Path
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.distributed_study import validate_training_steps, validate_training_contract


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_metrics(reference, candidate):
    if not reference or not candidate:
        raise ValueError('不能以空参数集合宣称数值等价')
    import torch

    if reference.keys() != candidate.keys():
        raise ValueError('参数或梯度名称集合不一致')
    total = 0.0
    error = 0.0
    maximum = 0.0
    changed = 0
    elements = 0
    for name, a in reference.items():
        b = candidate[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('张量结构不同: ' + name)
        a, b = a.reshape(-1), b.reshape(-1)
        tensor_changed = False
        for start in range(0, a.numel(), 1048576):
            x = a[start : start + 1048576].double()
            y = b[start : start + 1048576].double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('非有限张量: ' + name)
            delta = x - y
            local = float(delta.abs().max()) if delta.numel() else 0.0
            maximum = max(maximum, local)
            error += float(torch.dot(delta, delta))
            total += float(torch.dot(x, x))
            tensor_changed = tensor_changed or local != 0
            elements += x.numel()
        changed += tensor_changed
    if elements == 0:
        raise ValueError('参数集合没有元素')
    return {
        'relative_l2': (error / max(total, 1e-30)) ** 0.5,
        'max_abs': maximum,
        'changed_tensors': changed,
        'elements': elements,
    }


def main(args):
    import torch

    torch.set_num_threads(4)
    reference = Path(args.reference)
    candidate = Path(args.candidate)
    a, b = read(reference / 'summary.json'), read(candidate / 'summary.json')
    if not a['completed'] or not b['completed']:
        raise ValueError('尝试没有完成')
    left, right = a['contract'], b['contract']
    for key in [
        'plan_sha256',
        'model_digest',
        'phase',
        'optimizer',
        'precision',
        'global_batch',
        'sequence_length',
        'planned_steps',
    ]:
        if left[key] != right[key]:
            raise ValueError('训练合同差异: ' + key)
    if left['profile_enabled'] or right['profile_enabled']:
        raise ValueError('剖析运行不能代替数值对照')
    plan = read(args.plan)
    if file_hash(args.plan) != left['plan_sha256']:
        raise ValueError('计划来源不同')
    phases = plan[left['phase']]
    lrows = [
        json.loads(s) for s in (reference / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    rrows = [
        json.loads(s) for s in (candidate / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    records = read(Path(args.plan).parent / 'samples.json')
    if file_hash(Path(args.plan).parent / 'samples.json') != plan['samples_sha256']:
        raise ValueError('样本来源不符')
    for folder, summary, rows in [(reference, a, lrows), (candidate, b, rrows)]:
        start = read(folder / 'resume.json')['next_step'] if summary['resumed'] else 0
        initial = read(folder / 'input.json')
        effective = validate_training_contract(
            summary['contract'], initial['config'], plan, file_hash(args.plan)
        )
        if (
            summary['steps_completed'] != summary['contract']['planned_steps']
            or summary['steps_this_run'] != len(rows)
            or initial['contract'] != summary['contract']
        ):
            raise ValueError('训练终点或步数不符')
        validate_training_steps(
            rows,
            summary['contract'],
            records,
            rank=summary['rank'],
            start_step=start,
            warmup=effective.get('warmup_steps', 0),
        )
    # 恢复只比较共同的后续步，最终状态必须覆盖同一完整终点。
    by_step = {r['step']: r for r in lrows}
    loss_errors = []
    for row in rrows:
        prior = by_step[row['step']]
        if (
            row['global_sample_ids'] != prior['global_sample_ids']
            or row['global_valid_tokens'] != prior['global_valid_tokens']
        ):
            raise ValueError('训练样本或全局分母不同')
        loss_errors.append(abs(row['loss'] - prior['loss']) / max(abs(prior['loss']), 1e-12))
    if a['steps_completed'] != b['steps_completed']:
        raise ValueError('训练终点不一致')
    state_sha = {
        label: file_hash(folder / 'final-state.pt')
        for label, folder in [('reference', reference), ('candidate', candidate)]
    }
    state_a = torch.load(reference / 'final-state.pt', map_location='cpu', weights_only=True)
    state_b = torch.load(candidate / 'final-state.pt', map_location='cpu', weights_only=True)
    parameters = tensor_metrics(state_a, state_b)
    del state_a, state_b
    if state_sha != {
        label: file_hash(folder / 'final-state.pt')
        for label, folder in [('reference', reference), ('candidate', candidate)]
    }:
        raise ValueError('读取比较期间最终参数文件变化')
    gradient = None
    if (reference / 'final-gradients.pt').exists() and (candidate / 'final-gradients.pt').exists():
        ga = torch.load(reference / 'final-gradients.pt', map_location='cpu', weights_only=True)
        gb = torch.load(candidate / 'final-gradients.pt', map_location='cpu', weights_only=True)
        gradient = tensor_metrics(ga, gb)
        del ga, gb
    passed = (
        max(loss_errors) <= phases['loss_relative_tolerance']
        and parameters['relative_l2'] <= phases['final_parameter_relative_l2_tolerance']
        and parameters['max_abs'] <= phases['final_parameter_max_abs_tolerance']
    )
    result = {
        'passed': passed,
        'reference': str(reference),
        'candidate': str(candidate),
        'parameters': parameters,
        'last_gradient_diagnostic': gradient,
        'max_loss_relative_error': max(loss_errors),
        'compared_steps': [r['step'] for r in rrows],
        'final_step': a['steps_completed'],
        'plan_sha256': file_hash(args.plan),
        'state_sha256': state_sha,
        'summary_sha256': {
            'reference': file_hash(reference / 'summary.json'),
            'candidate': file_hash(candidate / 'summary.json'),
        },
        'steps_sha256': {
            'reference': file_hash(reference / 'steps.jsonl'),
            'candidate': file_hash(candidate / 'steps.jsonl'),
        },
        'input_sha256': {
            'reference': file_hash(reference / 'input.json'),
            'candidate': file_hash(candidate / 'input.json'),
        },
        'source_sha256': file_hash(__file__),
        'scope': 'declared numerical tolerances; not bitwise equality or application qualification',
    }
    write_once(args.output, result)
    print(json.dumps(result))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ['reference', 'candidate', 'plan', 'output']:
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    if not main(a):
        raise SystemExit(2)
