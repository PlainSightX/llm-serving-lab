"""冷启动三条件使用同一执行者；固定示范按明示预算进入实际上下文。"""

from .standard_actor import StandardActor
from .task_protocol import digest


class ColdStartActor(StandardActor):
    def __init__(self, examples=None):
        self.examples = examples

    def prepare(self, initial, view, trace, tokenize, max_input=3500):
        if self.examples is None:
            return super().prepare(initial, view, trace, tokenize, max_input)
        example_message = {
            'role': 'user',
            'content': 'Fixed public training demonstrations from other instances:\n'
            + self.examples,
        }
        budget = max_input
        while True:
            result = super().prepare(initial, view, trace, tokenize, budget)
            messages = result['messages'][:1] + [example_message] + result['messages'][1:]
            ids = tokenize(messages)
            if len(ids) <= max_input:
                break
            budget -= len(ids) - max_input
            if budget <= 0:
                raise ValueError('固定示范与当前任务不能在预算内共存')
        result['messages'], result['input_token_ids'] = messages, ids
        result['context'].update(
            actor='public_choice_fixed_demonstrations_v1',
            fixed_examples_digest=digest(self.examples),
            input_tokens=len(ids),
            max_input_tokens=max_input,
            remaining_own_context_budget=budget,
        )
        return result
