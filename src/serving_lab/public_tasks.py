"""Q2公开信息执行：actor无任务路径、隐藏facts或成功判定读取权。"""

from __future__ import annotations
import copy
import re
from .task_protocol import public_view, canonical, digest, TOOL_CONTRACT


class PublicEpisode:
    def __init__(self, env, max_steps, *, stagnation_limit=None):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('无效步数')
        if stagnation_limit is not None and (
            type(stagnation_limit) is not int or stagnation_limit < 2
        ):
            raise ValueError('无效停滞阈值')
        self.env = env
        self.max_steps = max_steps
        self.trace = []
        self.started = False
        self.terminal = False
        self.done = False
        self.stagnation_limit = stagnation_limit
        self.stagnation_count = 0
        self.previous_action = None

    def reset(self):
        if self.started:
            raise ValueError('禁止重复reset')
        self.state = self.env.reset()
        self.started = True
        if self.state.get('won'):
            raise ValueError('初态已成功')
        self.initial = public_view(self.state)
        self.visible = copy.deepcopy(self.initial)
        return copy.deepcopy(self.visible)

    def step(self, action):
        if not self.started or self.terminal:
            raise ValueError('禁止未开始或终结后动作')
        before = copy.deepcopy(self.visible)
        valid = (
            isinstance(action, str)
            and bool(action.strip())
            and len(action) <= 256
            and '\n' not in action
            and '\r' not in action
        )
        allowed = valid and (
            action in before['admissible_commands'] or action in {'look', 'inventory'}
        )
        if allowed:
            self.state, _, done = self.env.step(action)
            self.done = bool(done)
            after = public_view(self.state)
        else:
            after = {
                **before,
                'feedback': 'Action rejected: return exactly one available command, look, or inventory.',
            }
        self.trace.append(
            {
                'step': len(self.trace) + 1,
                'before': before,
                'action': action,
                'format_valid': valid,
                'admissible': bool(allowed),
                'after': after,
            }
        )
        signature = lambda v: (v['feedback'], tuple(sorted(v['admissible_commands'])))
        if signature(before) == signature(after):
            self.stagnation_count = (
                self.stagnation_count + 1 if action == self.previous_action else 1
            )
        else:
            self.stagnation_count = 0
        self.previous_action = action
        self.visible = copy.deepcopy(after)
        self.terminal = (
            self.done
            or bool(self.state.get('won'))
            or len(self.trace) >= self.max_steps
            or bool(self.stagnation_limit and self.stagnation_count >= self.stagnation_limit)
        )
        return copy.deepcopy(after)

    def result(self):
        if not self.terminal:
            raise ValueError('终结前不开放评价')
        n = len(self.trace)
        success = bool(self.state.get('won'))
        stopped = bool(self.stagnation_limit and self.stagnation_count >= self.stagnation_limit)
        return {
            'success': success,
            'steps': n,
            'max_steps': self.max_steps,
            'termination': 'won'
            if success
            else (
                'environment_done'
                if self.done
                else ('stagnation_terminated' if stopped else 'step_budget')
            ),
            'format_valid_rate': sum(t['format_valid'] for t in self.trace) / n,
            'admissible_rate': sum(t['admissible'] for t in self.trace) / n,
        }


class PublicPolicy:
    """从公开目标和动作集合搜索；没有任务family或位置答案输入。"""

    def __init__(self):
        self.goal = None
        self.visited = set()
        self.opened = set()
        self.held = None
        self.processed = set()
        self.placed = set()
        self.last = None
        self.known_objects = {}
        self.current_place = None
        self.known_lamps = {}
        self.lamp_type = None

    def choose(self, view):
        view = public_view(view)
        text = view['feedback']
        commands = sorted(view['admissible_commands'])
        if self.goal is None:
            match = re.search(r'Your task is to:\s*(.+)', text, re.I)
            if not match:
                raise ValueError('公开目标不可解析')
            self.goal = match.group(1).strip().rstrip('.')
            g = self.goal.lower()
            self.process = next(
                (
                    v
                    for v, pattern in (
                        ('clean', r'\bclean\b'),
                        ('heat', r'\b(?:heat|hot)\b'),
                        ('cool', r'\b(?:cool|cold)\b'),
                    )
                    if re.search(pattern, g)
                ),
                None,
            )
            self.examine = 'examine' in g or g.startswith('look at ')
            self.two = bool(re.search(r'\btwo\b', g))
            objects = re.findall(
                r'^(?:put|examine|clean|heat|cool|find)\s+(?:a|an|the|two|some)\s+(?:(?:clean|hot|cool|cold)\s+)?([a-z]+)',
                g,
            )
            # clean/heat/cool也可能是后置修饰，取put/examine后的名词。
            self.object = objects[0] if objects else ''
            look_object = re.match(r'look at\s+(?:(?:a|an|the)\s+)?([a-z]+)\s+under\b', g)
            if look_object:
                self.object = look_object[1]
            lamp = re.search(r'\b(?:with|under)\s+(?:(?:a|an|the)\s+)?([a-z]*lamp)\b', g)
            self.lamp_type = lamp[1] if lamp else None
            target = re.search(r'\b(?:in/on|in|on)\s+(?:(?:a|an|the)\s+)?([a-z]+)', g)
            self.target = target.group(1) if target else None
        for command in commands:
            match = re.fullmatch(r'take (.+) from (.+)', command)
            if match and match[1].startswith(self.object + ' '):
                self.known_objects[match[1]] = match[2]
        if self.last:
            if self.last.startswith('go to ') and text.startswith(
                'You arrive at ' + self.last[6:] + '.'
            ):
                self.current_place = self.last[6:]
            if self.last.startswith('take ') and re.search(r'You (?:pick up|take)', text, re.I):
                self.held = self.last[5:].split(' from ')[0]
            normalized = re.sub(r'\bthe\s+', '', text.lower()).strip()
            if (
                self.held
                and self.last.startswith(('clean ', 'heat ', 'cool '))
                and normalized.startswith('you ' + self.last.split()[0] + ' ' + self.held + ' ')
            ):
                self.processed.add(self.held)
            if (
                self.held
                and self.last.startswith(('put ', 'move '))
                and normalized.startswith(
                    ('you put ' + self.held + ' ', 'you move ' + self.held + ' ')
                )
            ):
                self.placed.add(self.held)
                self.held = None
        # 灯的位置只来自已访问地点的公开use候选；找到物品后可以返回。
        for command in commands:
            if (
                self.current_place
                and command.startswith('use ')
                and (
                    command.startswith('use ' + self.lamp_type + ' ')
                    if self.lamp_type
                    else 'lamp' in command
                )
            ):
                self.known_lamps[command] = self.current_place

        def choose(command):
            self.last = command
            return command

        if self.held:
            if self.examine:
                uses = [
                    c
                    for c in commands
                    if c.startswith('use ')
                    and (
                        c.startswith('use ' + self.lamp_type + ' ')
                        if self.lamp_type
                        else 'lamp' in c
                    )
                ]
                if uses:
                    return choose(uses[0])
                returns = sorted(
                    {'go to ' + place for place in self.known_lamps.values()} & set(commands)
                )
                if returns:
                    return choose(returns[0])
            if self.process and self.held not in self.processed:
                operations = [
                    c for c in commands if c.startswith(self.process + ' ' + self.held + ' ')
                ]
                if operations:
                    return choose(operations[0])
                destination = {'clean': 'sinkbasin', 'heat': 'microwave', 'cool': 'fridge'}[
                    self.process
                ]
            else:
                puts = [
                    c
                    for c in commands
                    if c.startswith(
                        (
                            'put ' + self.held + ' in/on ' + (self.target or '!') + ' ',
                            'move ' + self.held + ' to ' + (self.target or '!') + ' ',
                        )
                    )
                ]
                if puts:
                    return choose(puts[0])
                destination = self.target
            opens = [
                c for c in commands if destination and c.startswith('open ' + destination + ' ')
            ]
            if opens:
                self.opened.add(opens[0])
                return choose(opens[0])
            moves = [
                c for c in commands if destination and c.startswith('go to ' + destination + ' ')
            ]
            if moves:
                return choose(moves[0])
        takes = [
            c
            for c in commands
            if c.startswith('take ' + self.object + ' ')
            and c[5:].split(' from ')[0] not in self.placed
        ]
        if takes and not self.held:
            return choose(takes[0])
        if not self.held:
            known = sorted(
                {
                    'go to ' + location
                    for obj, location in self.known_objects.items()
                    if obj not in self.placed
                }
                & set(commands)
            )
            if known:
                return choose(known[0])
        opens = [c for c in commands if c.startswith('open ') and c not in self.opened]
        if opens:
            self.opened.add(opens[0])
            return choose(opens[0])
        moves = [c for c in commands if c.startswith('go to ') and c not in self.visited]
        if moves:
            self.visited.add(moves[0])
            return choose(moves[0])
        return choose('look')


def action_context(
    initial, view, trace, history, tokenize, max_input, *, instruction=TOOL_CONTRACT
):
    """固定目标/当前观察/历史不裁剪；只删除最旧的自身完整动作反馈对。"""
    view = public_view(view)
    pairs = [
        {
            'action': t['action'],
            'feedback': t['after']['feedback'] if 'after' in t else t['feedback'],
        }
        for t in trace
    ]
    dropped = 0
    while True:
        messages = [
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': 'Initial public observation:\n' + initial},
        ]
        if history is not None:
            messages.append(
                {
                    'role': 'user',
                    'content': 'Past public successful experience from a different instance (may be partial):\n'
                    + history,
                }
            )
        if pairs:
            messages.append(
                {
                    'role': 'user',
                    'content': 'Your earlier actions and feedback:\n' + canonical(pairs),
                }
            )
        messages.append(
            {'role': 'user', 'content': 'Current public observation:\n' + canonical(view)}
        )
        ids = tokenize(messages)
        if len(ids) <= max_input:
            return messages, ids, {'dropped_steps': dropped, 'history_present': history is not None}
        if not pairs:
            raise ValueError('固定公开信息超过输入预算，禁止隐式截断或重试')
        pairs.pop(0)
        dropped += 1


def verified_history(record, *, cutoff):
    if (
        record.get('purpose') == 'actor_diagnostic'
        or record.get('eligible_for_primary_history') is False
    ):
        raise ValueError('开发诊断不能晋级为主要历史或训练数据')
    if (
        record.get('origin') != 'model_public'
        or record.get('role') != 'support'
        or record.get('sequence', cutoff) >= cutoff
        or record.get('result', {}).get('success') is not True
        or not record.get('trace')
        or any(
            not s.get('receipt', {}).get('protocol_valid') or not s['receipt'].get('release')
            for s in record['trace']
        )
    ):
        raise ValueError('共同历史需要本轮公开模型成功及每动作真实回执')
    return canonical(
        {
            'initial': record['initial']['feedback'],
            'trajectory': [
                {
                    'observation': s['before']['feedback'],
                    'action': s['action'],
                    'feedback': s['after']['feedback'],
                }
                for s in record['trace']
            ],
        }
    )


def matched_histories(left, right, tokenizer, initial, view, cap=768):
    encoded = [tokenizer.encode(t, add_special_tokens=False) for t in (left, right)]
    n = min(cap, *map(len, encoded))
    if n < 1:
        raise ValueError('空历史不配对')
    texts = [tokenizer.decode(ids[:n], skip_special_tokens=False) for ids in encoded]
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    lengths = [len(action_context(initial, view, [], t, tokenize, 3000)[1]) for t in texts]
    if (
        len({len(tokenizer.encode(t, add_special_tokens=False)) for t in texts}) != 1
        or len(set(lengths)) != 1
    ):
        raise ValueError('历史实际模板token预算不可比')
    return texts, n, lengths[0]
