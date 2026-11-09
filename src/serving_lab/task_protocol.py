"""Q1文本交互任务合同：公开观察与私有评价分开，不执行模型训练。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

FAMILIES = (
    'pick_and_place_simple',
    'look_at_obj_in_light',
    'pick_clean_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_two_obj_and_place',
)
TOOL_CONTRACT = (
    'Choose exactly one command for the current text environment. Use the current feedback and available commands. '
    'look and inventory are also allowed. Past demonstrations describe other instances; their object IDs and locations '
    'may differ. Return only one command, without explanation or multiple actions.'
)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_manifest(manifest):
    """Q1切片不是任意任务数组；完整角色/关系是执行前条件。"""
    tasks = manifest.get('tasks')
    if (
        manifest.get('name') != 'q1-alfworld-text-qualification-v1'
        or not isinstance(tasks, list)
        or len(tasks) != 18
    ):
        raise ValueError('Q1必须含6个support、6个target及6个保护身份')
    if type(manifest.get('max_steps')) is not int or not 0 < manifest['max_steps'] <= 1000:
        raise ValueError('无效动作预算')
    by_id = {}
    sequences = set()
    keys = set()
    groups = {}
    for task in tasks:
        key = task.get('task_id')
        sequence = task.get('sequence')
        role = task.get('role')
        family = task.get('family')
        if (
            not isinstance(key, str)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', key)
            or key in by_id
            or type(sequence) is not int
            or sequence in sequences
            or role not in {'support', 'target', 'protected'}
            or family not in FAMILIES
            or not isinstance(task.get('source_key'), str)
            or not task['source_key']
            or task.get('source_key') in keys
        ):
            raise ValueError('任务身份/角色/顺序不符')
        by_id[key] = task
        sequences.add(sequence)
        keys.add(task.get('source_key'))
        if (family, role) in groups:
            raise ValueError('任务族角色重复')
        groups[family, role] = task
        if role == 'protected':
            if (
                task.get('extracted') is not False
                or task.get('source_split') not in {'valid_unseen', 'eval_out_of_distribution'}
                or 'game_path' in task
            ):
                raise ValueError('保护任务必须保持未提取身份')
        else:
            expected_files = {f'games/{key}/game.tw-pddl', f'games/{key}/traj_data.json'}
            if (
                task.get('source_split') != 'train'
                or set(task.get('files', {})) != expected_files
                or task.get('game_path') != f'games/{key}/game.tw-pddl'
            ):
                raise ValueError('可执行任务的路径或切分不符')
            if not all(
                isinstance(h, str) and re.fullmatch(r'[a-f0-9]{64}', h)
                for h in task['files'].values()
            ):
                raise ValueError('文件摘要无效')
    for family in FAMILIES:
        if any((family, role) not in groups for role in ('support', 'target', 'protected')):
            raise ValueError('任务族覆盖不完整')
        source = groups[family, 'support']
        target = groups[family, 'target']
        if (
            target.get('candidate_source_task_id') != source['task_id']
            or source['sequence'] >= target['sequence']
            or source['scene'] == target['scene']
            or source['parameter_digest'] == target['parameter_digest']
        ):
            raise ValueError('候选复用关系不符')
    if [t['sequence'] for t in tasks] != sorted(sequences):
        raise ValueError('任务必须按冻结顺序执行')
    return by_id


def validate_record(record, expected_task):
    if (
        record.get('producer') != 'serving_lab.task_protocol.v1'
        or 'record_hash' not in record
        or digest({k: v for k, v in record.items() if k != 'record_hash'}) != record['record_hash']
        or canonical(record.get('task')) != canonical(expected_task)
    ):
        raise ValueError('episode记录摘要或冻结任务身份不符')
    if expected_task['role'] not in {'support', 'target'}:
        raise ValueError('不能接纳保护任务记录')
    result = record['result']
    trace = record['trace']
    if (
        not isinstance(trace, list)
        or not trace
        or type(result.get('success')) is not bool
        or result.get('steps') != len(trace)
        or type(result.get('max_steps')) is not int
        or not 0 < len(trace) <= result['max_steps']
        or (result['success'] != (result['termination'] == 'won'))
        or result['termination'] not in {'won', 'environment_done', 'step_budget'}
        or (result['termination'] == 'step_budget' and len(trace) != result['max_steps'])
        or type(record.get('feedback_available_sequence')) is not int
        or record['feedback_available_sequence'] < expected_task['sequence']
    ):
        raise ValueError('回合反馈/步数不符')
    views = [record['initial']]
    for i, step in enumerate(trace, 1):
        if (
            step['step'] != i
            or type(step['format_valid']) is not bool
            or type(step['admissible']) is not bool
        ):
            raise ValueError('轨迹结构不符')
        views.extend([step['before'], step['after']])
    for view in views:
        if set(view) != {'feedback', 'admissible_commands'}:
            raise ValueError('轨迹公开观察夹带私有字段')
        public_view(view)
    if result['format_valid_rate'] != sum(t['format_valid'] for t in trace) / len(trace) or result[
        'admissible_rate'
    ] != sum(t['admissible'] for t in trace) / len(trace):
        raise ValueError('分项指标不符')
    return record


def checked_target(root, member):
    """只允许归档中的相对文件，不能让源数据路径越过本任务目录。"""
    normalized = member.replace('\\', '/')
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts or ':' in normalized:
        raise ValueError('归档路径越界')
    target = (Path(root) / Path(*path.parts)).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('归档路径越界')
    return target


def public_view(state):
    feedback = state.get('feedback')
    commands = state.get('admissible_commands')
    if (
        not isinstance(feedback, str)
        or not isinstance(commands, list)
        or not all(isinstance(c, str) for c in commands)
    ):
        raise ValueError('环境缺少公开文本观察')
    return {'feedback': feedback, 'admissible_commands': list(commands)}


class Episode:
    def __init__(self, task, env, *, max_steps, origin, reference_method=None):
        if task['role'] == 'protected':
            raise ValueError('Q1禁止运行保护任务')
        if (
            task['role'] not in {'support', 'target'}
            or type(max_steps) is not int
            or max_steps <= 0
        ):
            raise ValueError('无效任务或预算')
        if origin not in {
            'privileged_reference',
            'constant_look',
            'literal_replay',
            'trajectory_counterfactual',
            'scripted_control',
        }:
            raise ValueError('Q1尚未接入真实模型回执，不能声明模型来源')
        if origin == 'privileged_reference' and reference_method not in {'handcoded', 'planner'}:
            raise ValueError('参考演示必须注明具体特权策略')
        self.task = copy.deepcopy(task)
        self.env = env
        self.max_steps = max_steps
        self.origin = origin
        self.reference_method = reference_method
        self.started = False
        self.closed = False
        self.terminal = False
        self.trace = []
        self.state = None

    def reset(self):
        if self.started or self.closed:
            raise ValueError('一次episode只能reset一次')
        self.state = self.env.reset()
        self.started = True
        if bool(self.state.get('won')):
            raise ValueError('任务初态已完成，不能用于本次前检')
        self.initial = public_view(self.state)
        self.visible = copy.deepcopy(self.initial)
        self.done = False
        return copy.deepcopy(self.initial)

    def reference_action(self):
        if (
            self.origin != 'privileged_reference'
            or not self.started
            or self.terminal
            or self.closed
        ):
            raise ValueError('只有参考策略可读取特权动作')
        plan = self.state.get('extra.expert_plan')
        if not isinstance(plan, list) or not plan or not isinstance(plan[0], str):
            raise ValueError('参考策略没有动作')
        return plan[0]

    def step(self, action):
        if not self.started or self.terminal or self.closed:
            raise ValueError('禁止未开始或结束后执行动作')
        before = copy.deepcopy(self.visible)
        valid = (
            isinstance(action, str)
            and bool(action.strip())
            and len(action) <= 256
            and '\n' not in action
            and '\r' not in action
        )
        admissible = valid and (
            action in before['admissible_commands'] or action in {'look', 'inventory'}
        )
        if valid:
            self.state, _, done = self.env.step(action)
            self.done = bool(done)
            after = public_view(self.state)
        else:
            # 格式错误同样消耗一步；不通过丢弃坏输出获得隐形重试。
            after = {
                **before,
                'feedback': 'Action rejected: supply one nonempty command of at most 256 characters.',
            }
        self.trace.append(
            {
                'step': len(self.trace) + 1,
                'before': before,
                'action': action[:256] if isinstance(action, str) else None,
                'format_valid': valid,
                'admissible': admissible,
                'after': after,
            }
        )
        self.visible = copy.deepcopy(after)
        self.terminal = (
            self.done or bool(self.state.get('won')) or len(self.trace) >= self.max_steps
        )
        return copy.deepcopy(after)

    def result(self):
        if not self.started or not self.terminal:
            raise ValueError('执行终结前不开放评价反馈')
        success = bool(self.state.get('won'))
        return {
            'success': success,
            'steps': len(self.trace),
            'max_steps': self.max_steps,
            'termination': 'won'
            if success
            else ('environment_done' if self.done else 'step_budget'),
            'format_valid_rate': sum(t['format_valid'] for t in self.trace) / len(self.trace)
            if self.trace
            else 0.0,
            'admissible_rate': sum(t['admissible'] for t in self.trace) / len(self.trace)
            if self.trace
            else 0.0,
        }

    def record(self):
        result = self.result()
        record = {
            'producer': 'serving_lab.task_protocol.v1',
            'task': self.task,
            'origin': self.origin,
            'reference_method': self.reference_method,
            'initial': self.initial,
            'trace': self.trace,
            'result': result,
            'feedback_available_sequence': self.task['sequence'],
        }
        return {**copy.deepcopy(record), 'record_hash': digest(record)}

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True


def history_material(record, *, cutoff, purpose='diagnostic'):
    """当前只接纳成功的参考演示；Q2模型经历必须另接真实执行来源。"""
    if purpose != 'diagnostic':
        raise ValueError('特权参考只供诊断，不接纳为主要比较或参数训练历史')
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError('历史截止点必须是非负整数')
    task = record['task']
    result = record['result']
    validate_record(record, task)
    if record.get('producer') != 'serving_lab.task_protocol.v1' or 'record_hash' not in record:
        raise ValueError('缺少受控episode记录身份')
    if (
        task['role'] != 'support'
        or task['sequence'] >= cutoff
        or record['feedback_available_sequence'] >= cutoff
        or record['origin'] != 'privileged_reference'
        or record.get('reference_method') not in {'handcoded', 'planner'}
        or result['success'] is not True
        or result['termination'] != 'won'
    ):
        raise ValueError('历史不符合来源、时间、切分或完成条件')
    if (
        'record_hash' in record
        and digest({k: v for k, v in record.items() if k != 'record_hash'}) != record['record_hash']
    ):
        raise ValueError('历史记录已改变')
    public_trace = [
        {
            'observation': t['before']['feedback'],
            'action': t['action'],
            'feedback': t['after']['feedback'],
        }
        for t in record['trace']
    ]
    material = {
        'source_task_id': task['task_id'],
        'source_record_hash': record.get('record_hash'),
        'producer': 'serving_lab.diagnostic_history.v2',
        'source_role': task['role'],
        'source_sequence': task['sequence'],
        'feedback_available_sequence': record['feedback_available_sequence'],
        'cutoff_sequence': cutoff,
        'origin': 'privileged_reference',
        'reference_method': record.get('reference_method'),
        'initial_observation': record['initial']['feedback'],
        'trajectory': public_trace,
        'verified_success': True,
        'eligible_for_parameter_training': False,
        'history_role': 'oracle_diagnostic_only',
        'eligible_for_primary_comparison_history': False,
        'token_budget_status': 'must_bind_actual_model_tokenizer_before_model_trial',
    }
    return {**material, 'material_hash': digest(material)}


def build_context(observation, history, *, condition, cutoff, purpose='diagnostic'):
    if purpose != 'diagnostic':
        raise ValueError('Q1上下文尚不具备主要比较资格')
    if condition not in {'none', 'oracle_related', 'different_family_reference'}:
        raise ValueError('未知诊断条件')
    if (condition == 'none') != (history is None):
        raise ValueError('条件与历史不符')
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError('无效上下文截止点')
    messages = [{'role': 'system', 'content': TOOL_CONTRACT}]
    if history is not None:
        if (
            history.get('producer') != 'serving_lab.diagnostic_history.v2'
            or 'material_hash' not in history
            or digest({k: v for k, v in history.items() if k != 'material_hash'})
            != history['material_hash']
            or history.get('source_role') != 'support'
            or history.get('origin') != 'privileged_reference'
            or history.get('reference_method') not in {'handcoded', 'planner'}
            or history.get('verified_success') is not True
            or history.get('history_role') != 'oracle_diagnostic_only'
            or history.get('eligible_for_parameter_training') is not False
            or history.get('eligible_for_primary_comparison_history') is not False
            or history.get('cutoff_sequence') != cutoff
            or type(history.get('source_sequence')) is not int
            or history['source_sequence'] >= cutoff
            or type(history.get('feedback_available_sequence')) is not int
            or history['feedback_available_sequence'] >= cutoff
        ):
            raise ValueError('上下文只接受当前截止点已校验的诊断历史')
        material = {k: history[k] for k in ('initial_observation', 'trajectory')}
        messages.append(
            {
                'role': 'user',
                'content': 'Past verified demonstration (a different instance):\n'
                + canonical(material),
            }
        )
    messages.append(
        {'role': 'user', 'content': 'Current observation:\n' + canonical(public_view(observation))}
    )
    return {
        'messages': messages,
        'metadata': {
            'condition': condition,
            'cutoff_sequence': cutoff,
            'model_trial_ready': False,
            'source_ids': [history['source_task_id']] if history else [],
            'source_record_hashes': [history['source_record_hash']] if history else [],
            'history_origin': history['origin'] if history else None,
            'history_reference_method': history.get('reference_method') if history else None,
            'history_role': 'oracle_diagnostic_only' if history else 'none',
            'eligible_for_primary_comparison_history': False,
            'eligible_for_parameter_training': False,
            'history_characters': len(messages[1]['content']) if history else 0,
            'budget_status': 'must bind model/tokenizer, full context budget and token-matched controls before inference',
        },
    }


def verify_files(task, data_root):
    for name, expected in task['files'].items():
        path = checked_target(data_root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('任务源文件与冻结清单不符')


def open_environment(task, data_root, *, reference=False, reference_method='handcoded'):
    if task['role'] == 'protected':
        raise ValueError('Q1禁止reset保护任务')
    verify_files(task, data_root)
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
    )

    wrappers = [AlfredDemangler()]
    extras = []
    if reference:
        if reference_method not in {'handcoded', 'planner'}:
            raise ValueError('未知参考策略')
        expert_type = (
            AlfredExpertType.HANDCODED
            if reference_method == 'handcoded'
            else AlfredExpertType.PLANNER
        )
        wrappers.append(AlfredExpert(expert_type=expert_type))
        extras = ['expert_plan']
    return textworld.start(
        str(checked_target(data_root, task['game_path'])),
        request_infos=textworld.EnvInfos(won=True, admissible_commands=True, extras=extras),
        wrappers=wrappers,
    )
