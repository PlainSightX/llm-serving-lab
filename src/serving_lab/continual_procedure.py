"""N5监督式流程知识巩固：完整公告是历史来源，查询只携带流程身份。"""

import json
from .standard_actor import INSTRUCTION, StandardActor
from .task_protocol import canonical, digest
from .workflow import WorkflowActor, workflow_spec

ACTOR_IDENTITIES = {
    'C0-Qwen3-1.7B': (
        '1ee83b8562404b56bdb208a675bb5d4e4c83b827156df8f6b1e73d30842989de',
        '040bf1295cf8a6198df7888534b313ae3bc049b8c2c715bf81bf4579669e37bd',
    ),
    'Qwen3-4B-fixed-original': (
        'b45fba1ab01819bd9540b73858cd32e109bfeefed07ac36b5ada9248650dbba2',
        '3136b44de9201303ec3dcfd0ed66595c94c15ff4f58862f2aff474ade48e788c',
    ),
    # 单次447步基础适配及独立adapter重建已核，只允许先进入资格，不继承应用/R结论。
    'Qwen3-4B-shared-basic': (
        '799f0cacc88f3ed433f591970812b35694d306577bc5340464ee0a49266ea8cc',
        '98d39677740ed06dc3c379193d0d86834b66d5d2dc0f7c1c97858db03321946c',
    ),
}


def validate_actor_binding(role, manifest, binding):
    if role not in ACTOR_IDENTITIES:
        raise ValueError('不是已冻结共同actor候选')
    artifact, parameters = ACTOR_IDENTITIES[role]
    if (
        manifest['artifact_digest'] != artifact
        or manifest['parameter_digest'] != parameters
        or binding['artifact_digest'] != artifact
        or binding['parameter_digest'] != parameters
        or manifest['profile'] != binding['profile']
        or manifest['profile_digest'] != digest(binding['profile'])
    ):
        raise ValueError('共同actor的实际参数或完整profile不同')
    return role


def validate_memory_snapshot(snapshot, workflow, stage):
    if (
        snapshot['stage'] > stage
        or snapshot['source_role'] != 'train'
        or snapshot['workflow'] != workflow
        or not snapshot['entry_digests']
        or not snapshot['prompt']
    ):
        raise ValueError('记忆不属于当前阶段的同版本训练来源')
    return snapshot


def qualification_cases(tasks):
    return [
        {
            'id': 'n5-' + task['task_id'] + '-' + condition,
            'source_task_id': task['task_id'],
            'condition': condition,
            'phase': task['phase'],
            'role': 'dev',
            'base_id': task['base_id'],
            'thermal': task['workflow']['thermal'],
        }
        for task in tasks.values()
        if task['role'] == 'dev'
        for condition in ['explicit', 'memory']
    ]


def validate_qualification_cases(cases, tasks):
    expected = qualification_cases(tasks)
    if len(expected) != 36 or cases != expected:
        raise ValueError('资格必须恰好覆盖18个实际dev任务及两种条件，顺序和元数据不得变化')
    return cases


class ProcedureActor(StandardActor):
    def __init__(self, workflow, memory=None, *, explicit=False):
        if workflow != workflow_spec(
            workflow['name'] + '-v' + str(workflow['revision']), workflow['thermal']
        ):
            raise ValueError('非冻结流程')
        self.workflow = workflow
        self.memory = memory or []
        self.explicit = explicit

    def prepare(self, initial, view, trace, tokenize, max_input=3500):
        if self.explicit:
            return WorkflowActor(self.workflow, self.memory).prepare(
                initial, view, trace, tokenize, max_input
            )
        identity = {k: self.workflow[k] for k in ['name', 'revision', 'thermal']}

        def augment(messages):
            document = json.loads(messages[1]['content'])
            document['PROCEDURE'] = identity
            document['AVAILABLE_PROCEDURE_MEMORY'] = self.memory
            instruction = (
                INSTRUCTION
                + ' Task success requires both the household goal and the named PROCEDURE revision. '
                'The procedure was announced in earlier work; its full steps are not repeated in this query. '
                'Use learned procedure knowledge or AVAILABLE_PROCEDURE_MEMORY. Follow the requested revision. '
                'Memory examples come from other instances: adapt to current objects, places, and commands.'
            )
            return [
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': canonical(document)},
            ]

        prepared = super().prepare(
            initial, view, trace, lambda messages: tokenize(augment(messages)), max_input
        )
        prepared['messages'] = augment(prepared['messages'])
        if tokenize(prepared['messages']) != prepared['input_token_ids']:
            raise ValueError('实际模板token与冻结消息不符')
        prepared['context'].update(
            actor='n5_announced_procedure_query_v1',
            procedure=identity,
            procedure_digest=digest(identity),
            memory_digest=digest(self.memory),
            full_rule_in_query=False,
        )
        return prepared
