"""构造显式控制面夹具，验证失败、发布和回滚；不模拟模型学习。"""

import json
import struct
from pathlib import Path

from .lab import Lab, LabError, canonical
from .reporting import write_report


def tasks():
    result = []
    for i in range(1, 13):
        version = 'v1' if i % 2 else 'v2'
        actions = (
            ['check_units', 'compare']
            if version == 'v1'
            else ['check_units', 'check_coverage', 'compare']
        )
        result.append(
            {
                'task_id': f'task_{i}',
                'sequence': i,
                'family': f'fixture_family_{i}',
                'split': 'train' if i <= 8 else 'holdout',
                'rule_version': version,
                'messages': [
                    {
                        'role': 'system',
                        'content': '输出JSON操作计划，仅包含actions字段。流程v1先校核单位再比较；v2还必须在比较前检查覆盖率。',
                    },
                    {
                        'role': 'user',
                        'content': f'公开协议样例{i}：机组{i}的实验比较，当前流程{version}。',
                    },
                ],
                'expected': {'actions': actions},
            }
        )
    return result


def fixture_adapter(path, number):
    path.mkdir()
    (path / 'adapter_config.json').write_text(
        canonical(
            {'peft_type': 'LORA', 'base_model_name_or_path': 'control-plane-fixture', 'r': 1}
        ),
        encoding='utf-8',
    )
    header = canonical({'fixture': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (path / 'adapter_model.safetensors').write_bytes(
        struct.pack('<Q', len(header)) + header + struct.pack('<f', number)
    )
    (path / 'FIXTURE_ONLY.txt').write_text(
        '只用于模型版本控制面的字节身份测试；不可作为真实LoRA加载。', encoding='utf-8'
    )


def run_demo(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise LabError('演示目录已存在，请指定新目录以保留上次结果')
    destination.mkdir(parents=True)
    data = tasks()
    (destination / 'tasks.jsonl').write_text(
        ''.join(canonical(t) + '\n' for t in data), encoding='utf-8'
    )
    with Lab(destination / 'workspace') as lab:
        lab.import_tasks(data)
        for t in data[:8]:
            lab.record('learn_' + t['task_id'], t['task_id'], 'baseline', t['expected'])
            lab.feedback('learn_' + t['task_id'])
        exported = lab.export_training(destination / 'training_export', cutoff=8)
        panel = [t['task_id'] for t in data[8:]]
        for model, number in [('fixture_v1', 1), ('fixture_bad', 2), ('fixture_v2', 3)]:
            path = destination / model
            fixture_adapter(path, number)
            lab.register(model, path, fixture=True)
            for t in data[8:]:
                lab.record(
                    model + '_' + t['task_id'],
                    t['task_id'],
                    model,
                    {} if model == 'fixture_bad' else t['expected'],
                )
            lab.evaluate(model + '_eval', model, panel)
        lab.promote('fixture_v1_eval')
        try:
            lab.promote('fixture_bad_eval', expected_active='fixture_v1')
        except LabError:
            rejected = True
        else:
            raise AssertionError('错误候选不应通过采用门')
        lab.promote('fixture_v2_eval', expected_active='fixture_v1')
        lab.rollback('fixture_v1', expected_active='fixture_v2')
        summary = lab.summary()
        summary['demo'] = {
            'evidence_level': 'control_plane',
            'bad_candidate_rejected': rejected,
            'rollback_verified': summary['active'] == 'fixture_v1',
            'training_examples': exported['count'],
            'fixture_notice': '结果由脚本产生，参数为格式夹具；无模型推理、真实训练或学习结论。',
        }
    report = write_report(summary, destination)
    return {
        'directory': str(destination),
        'report': str(report),
        'evidence_level': 'control_plane',
        'checks': summary['demo'],
    }
