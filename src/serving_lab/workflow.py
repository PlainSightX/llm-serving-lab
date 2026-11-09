"""公开流程约束、同物品过程评价和独立研究actor。"""

import json
import re
from collections import Counter

from .public_tasks import PublicEpisode, PublicPolicy
from .standard_actor import INSTRUCTION, StandardActor
from .task_protocol import canonical, digest

VARIANTS = ('preparation-v1', 'inspection-v1', 'preparation-v2')
THERMAL_FAMILIES = {
    'pick_heat_then_place_in_recep': 'heat',
    'pick_cool_then_place_in_recep': 'cool',
}
BASE_COUNTS = {'train': 24, 'dev': 6, 'future': 12}


def workflow_spec(variant, thermal):
    if variant not in VARIANTS or thermal not in ('heat', 'cool'):
        raise ValueError('流程或热处理类型不在冻结合同内')
    operations = {
        'preparation-v1': ['clean', thermal],
        'inspection-v1': [thermal, 'inventory'],
        'preparation-v2': [thermal, 'clean'],
    }[variant]
    instruction = (
        'For the target object you finally place, perform these operations exactly once each, '
        'in this order, after taking it and before placing it: ' + ' -> '.join(operations) + '. '
        'Only successful operations count. Other navigation and observation commands are allowed. '
        'Inventory inspection must publicly show that you are carrying the target object. '
        'Complete the original household goal as well. This current revision overrides older procedures.'
    )
    name, revision = variant.rsplit('-v', 1)
    return {
        'name': name,
        'revision': int(revision),
        'thermal': thermal,
        'operations': operations,
        'instruction': instruction,
    }


def validate_workflow_manifest(manifest):
    if manifest.get('name') != 'q2-public-workflows-v1' or manifest.get('max_steps') != 100:
        raise ValueError('不是冻结的流程任务合同')
    contract = manifest['source_contract']
    if (
        contract.get('origin') != 'teacher_public_program'
        or digest(contract) != manifest['source_contract_digest']
    ):
        raise ValueError('流程教师来源合同不符')
    tasks = manifest['tasks']
    if len(tasks) != 126 or [task['sequence'] for task in tasks] != list(range(1, 127)):
        raise ValueError('必须冻结42个底层实例的三种流程，共126个任务身份')
    ids = {task['task_id']: task for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError('重复流程任务身份')
    counts = Counter((task['role'], task['variant']) for task in tasks)
    if counts != Counter(
        {(role, variant): count for role, count in BASE_COUNTS.items() for variant in VARIANTS}
    ):
        raise ValueError('流程/角色覆盖不完整')
    coverage = Counter((task['role'], task['variant'], task['family']) for task in tasks)
    if coverage != Counter(
        {
            (role, variant, family): count // 2
            for role, count in BASE_COUNTS.items()
            for variant in VARIANTS
            for family in THERMAL_FAMILIES
        }
    ):
        raise ValueError('热处理类型覆盖不完整')
    bases, base_variants, source_owners = {}, {}, {}
    for task in tasks:
        phase = (task['sequence'] - 1) // 42 + 1
        offset = (task['sequence'] - 1) % 42
        role = 'train' if offset < 24 else 'dev' if offset < 30 else 'future'
        if (
            type(task.get('phase')) is not int
            or task['phase'] != phase
            or task['variant'] != VARIANTS[phase - 1]
            or task['role'] != role
        ):
            raise ValueError('流程、阶段、角色与可用时间顺序不符')
        if task['family'] not in THERMAL_FAMILIES or task['workflow'] != workflow_spec(
            task['variant'], THERMAL_FAMILIES[task['family']]
        ):
            raise ValueError('流程文本、操作和任务热处理类型不一致')
        if (
            task['source_split'] != 'train'
            or not task['source_key'].startswith('train/')
            or task['source_key'] in manifest['excluded_source_keys']
        ):
            raise ValueError('来源越界或复用了旧任务实例')
        base = task['base_id']
        variants = base_variants.setdefault(base, set())
        if task['variant'] in variants:
            raise ValueError('同一底层实例重复流程或缺少另一流程')
        variants.add(task['variant'])
        previous_owner = source_owners.setdefault(task['source_key'], base)
        if previous_owner != base:
            raise ValueError('同一源实例被伪装为不同底层实例')
        expected = {f'games/{base}/game.tw-pddl', f'games/{base}/traj_data.json'}
        if set(task['files']) != expected or task['game_path'] != f'games/{base}/game.tw-pddl':
            raise ValueError('底层实例文件与身份不一致')
        signature = {
            key: task[key]
            for key in ('role', 'scene', 'family', 'source_key', 'parameter_digest', 'files')
        }
        if base in bases and bases[base] != signature:
            raise ValueError('同一底层实例被赋予不同来源或切分')
        bases[base] = signature
    if len(bases) != 42 or any(variants != set(VARIANTS) for variants in base_variants.values()):
        raise ValueError('底层实例数量不符')
    for key in ('scene', 'parameter_digest', 'source_key'):
        sets = {role: {task[key] for task in tasks if task['role'] == role} for role in BASE_COUNTS}
        if (
            sets['train'] & sets['dev']
            or sets['train'] & sets['future']
            or sets['dev'] & sets['future']
        ):
            raise ValueError('数据角色之间的来源隔离失败：' + key)
    if {task['scene'] for task in tasks if task['role'] == 'future'} != {14, 16, 25}:
        raise ValueError('未来评价场景变化')
    return ids


def observed_operations(trace, target):
    """只以公开成功反馈绑定同一个所持物品，失败动作不产生过程事实。"""
    held, operations = None, []
    for row in trace:
        action = row['action']
        feedback = re.sub(r'\bthe\s+', '', row['after']['feedback'].lower()).strip()
        take = re.fullmatch(r'take (.+) from (.+)', action)
        if take and feedback.startswith(
            ('you pick up ' + take[1] + ' ', 'you take ' + take[1] + ' ')
        ):
            held = take[1]
        process = re.fullmatch(r'(clean|heat|cool) (.+) with (.+)', action)
        if (
            process
            and held == target == process[2]
            and feedback.startswith('you ' + process[1] + ' ' + target + ' using ')
        ):
            operations.append(process[1])
        if (
            action == 'inventory'
            and held == target
            and feedback.startswith('you are carrying')
            and re.search(r'\b' + re.escape(target) + r'\b', feedback)
        ):
            operations.append('inventory')
        move = re.fullmatch(r'move (.+) to (.+)', action)
        if move and held == move[1] and feedback.startswith('you move ' + held + ' '):
            held = None
    return operations


def assess_workflow(trace, workflow, base_success):
    target = None
    for row in reversed(trace):
        match = re.fullmatch(r'move (.+) to (.+)', row['action'])
        normalized = re.sub(r'\bthe\s+', '', row['after']['feedback'].lower()).strip()
        if match and normalized.startswith('you move ' + match[1] + ' '):
            target = match[1]
            break
    observed = observed_operations(trace, target) if target else []
    required = workflow['operations']
    relevant = [operation for operation in observed if operation in required]
    passed = target is not None and relevant == required
    obsolete = (
        workflow['name'] == 'preparation'
        and workflow['revision'] == 2
        and relevant[:2] == ['clean', workflow['thermal']]
    )
    return {
        'base_success': bool(base_success),
        'workflow_passed': passed,
        'success': bool(base_success and passed),
        'placed_object': target,
        'operations_observed': relevant,
        'required_operations': required,
        'obsolete_order_observed': obsolete,
    }


class WorkflowEpisode(PublicEpisode):
    def __init__(self, env, max_steps, workflow, *, stagnation_limit=None):
        super().__init__(env, max_steps, stagnation_limit=stagnation_limit)
        self.workflow = workflow

    def result(self):
        result = super().result()
        assessment = assess_workflow(self.trace, self.workflow, result['success'])
        result['base_termination'] = result['termination']
        result.update(assessment)
        if result['base_success'] and not result['workflow_passed']:
            result['termination'] = 'workflow_failed'
        return result


class WorkflowPolicy:
    """复用公开搜索策略，仅按当前公开流程插入必要操作。"""

    def __init__(self, workflow):
        self.workflow = workflow
        self.base = PublicPolicy()

    def choose(self, view, trace):
        proposed = self.base.choose(view)
        target = self.base.held
        if target is None:
            return proposed
        required = self.workflow['operations']
        observed = [
            operation for operation in observed_operations(trace, target) if operation in required
        ]
        if observed != required[: len(observed)]:
            raise ValueError('程序策略已经违反当前公开顺序')
        if observed == required:
            return proposed
        operation = required[len(observed)]
        commands = sorted(view['admissible_commands'])
        if operation == 'inventory':
            chosen = 'inventory'
        else:
            direct = [
                command
                for command in commands
                if command.startswith(operation + ' ' + target + ' with ')
            ]
            tool = {'clean': 'sinkbasin', 'heat': 'microwave', 'cool': 'fridge'}[operation]
            opens = [command for command in commands if command.startswith('open ' + tool + ' ')]
            moves = [command for command in commands if command.startswith('go to ' + tool + ' ')]
            candidates = direct or opens or moves
            if not candidates:
                raise ValueError('当前公开命令中无法执行或到达流程操作')
            chosen = candidates[0]
        # 基础策略下一步读取实际执行动作的反馈，避免把被替换建议当成已执行。
        self.base.last = chosen
        return chosen


class WorkflowActor(StandardActor):
    def __init__(self, workflow, memory=None):
        self.workflow = workflow
        self.memory = memory or []

    def prepare(self, initial, view, trace, tokenize, max_input=3500):
        def augment(messages):
            document = json.loads(messages[1]['content'])
            document['GOAL'] += '\nWorkflow requirements: ' + self.workflow['instruction']
            document['CURRENT_WORKFLOW'] = {
                key: value for key, value in self.workflow.items() if key != 'instruction'
            }
            document['PAST_VERIFIED_EXPERIENCE'] = self.memory
            instruction = (
                INSTRUCTION + ' Task success requires satisfying both GOAL and CURRENT_WORKFLOW. '
                'Follow the current workflow even when the original household goal could be completed sooner. '
                'PAST_VERIFIED_EXPERIENCE contains examples from other instances. Reuse their procedures, '
                'but use the current goal, current objects and current commands instead of copying example identifiers.'
            )
            return [
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': canonical(document)},
            ]

        prepared = super().prepare(
            initial, view, trace, lambda messages: tokenize(augment(messages)), max_input
        )
        prepared['messages'] = augment(prepared['messages'])
        assert tokenize(prepared['messages']) == prepared['input_token_ids']
        prepared['context'].update(
            actor='public_workflow_goal_v3',
            workflow_digest=digest(self.workflow),
            memory_digest=digest(self.memory),
        )
        return prepared
