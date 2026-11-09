"""合成风电实验比较：模型选分析计划，受限执行器计算，独立参考聚合核验。"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

from .lab import LabError, canonical, digest

SOURCE_BYTES = Path(__file__).read_bytes()
FACTORS = ('mixed_units', 'shuffled_rows', 'missing_rows', 'distractor')
SYSTEM = '''你是风电研发分析助手。只输出一个JSON分析计划，不计算答案，不输出解释或Markdown。
格式：{"operation":"compare或abstain","candidate":"实验ID","reference":"实验ID","unit":"kW","support":"intersection","metric":"mae或rmse","gate":{"basis":"min_count或expected_fraction","threshold":数值}}
执行器将按时间戳连接观测与两个实验，先把MW乘1000转kW，在共同有效样本上计算误差，误差小者胜；不得按行号配对或分别计算不同样本。unit必须kW，support必须intersection。metric与实验ID由本次request指定，其他实验是无关数据。
你必须读取观测和两个指定实验的数据，按时间戳求三者交集；交集为空或不满足当前覆盖规则时operation用abstain，否则用compare。不能只复制request/rule，也不能因指标更好就跳过覆盖要求。'''


def validate_case(case):
    def require(condition):
        if not condition:
            raise LabError('业务数据条件不满足')

    try:
        require(set(case) == {'request', 'rule', 'expected_times', 'observations', 'runs'})
        require(set(case['request']) == {'candidate', 'reference', 'metric'})
        require(case['request']['metric'] in {'mae', 'rmse'})
        require(case['request']['candidate'] != case['request']['reference'])
        require(set(case['rule']) == {'version', 'minimum'})
        version, minimum = case['rule']['version'], case['rule']['minimum']
        require(
            version in {'v1', 'v2'} and type(minimum) in {int, float} and math.isfinite(minimum)
        )
        require(
            (version == 'v1' and type(minimum) is int and minimum >= 1)
            or (version == 'v2' and 0 < minimum <= 1)
        )
        times = case['expected_times']
        require(
            isinstance(times, list)
            and times
            and all(isinstance(t, str) and t for t in times)
            and len(times) == len(set(times))
        )
        require(
            isinstance(case['observations'], list)
            and case['observations']
            and isinstance(case['runs'], list)
            and 2 <= len(case['runs']) <= 10
        )
        ids = [r['id'] for r in case['runs']]
        require(len(ids) == len(set(ids)) and all(isinstance(x, str) and x for x in ids))
        require({case['request']['candidate'], case['request']['reference']} <= set(ids))
        for rows, field in [(case['observations'], 'kw')] + [
            (r['rows'], 'value') for r in case['runs']
        ]:
            require(isinstance(rows, list) and len(rows) == len({r['time'] for r in rows}))
            for row in rows:
                require(set(row) == {'time', field} and row['time'] in times)
                require(type(row[field]) in {int, float} and math.isfinite(row[field]))
        for run in case['runs']:
            require(set(run) == {'id', 'unit', 'rows'} and run['unit'] in {'kW', 'MW'})
        canonical(case)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise LabError('业务数据合同无效：单位、时间唯一性、数值或当次规则不完整') from exc


def reference_plan(case):
    return {
        **case['request'],
        'unit': 'kW',
        'support': 'intersection',
        'operation': 'compare' if oracle(case)['status'] == 'compared' else 'abstain',
        'gate': {
            'basis': 'min_count' if case['rule']['version'] == 'v1' else 'expected_fraction',
            'threshold': case['rule']['minimum'],
        },
    }


def messages(case):
    rule = (
        '当前规则v1：共同样本数不少于minimum，gate.basis用min_count。'
        if case['rule']['version'] == 'v1'
        else '当前规则v2：共同样本数除以expected_times数量不少于minimum，gate.basis用expected_fraction。'
    )
    return [
        {'role': 'system', 'content': SYSTEM + '\n' + rule},
        {'role': 'user', 'content': canonical(case)},
    ]


def execute_plan(case, plan):
    """有限计划语言，没有任意SQL、文件路径或代码执行能力。"""
    validate_case(case)
    if not isinstance(plan, dict) or set(plan) != {
        'operation',
        'candidate',
        'reference',
        'unit',
        'support',
        'metric',
        'gate',
    }:
        raise LabError('plan_schema')
    if (
        plan['unit'] != 'kW'
        or plan['support'] != 'intersection'
        or plan['metric'] not in {'mae', 'rmse'}
        or plan['operation'] not in {'compare', 'abstain'}
    ):
        raise LabError('unsafe_unit_support_or_metric')
    gate = plan['gate']
    if (
        not isinstance(gate, dict)
        or set(gate) != {'basis', 'threshold'}
        or gate['basis'] not in {'min_count', 'expected_fraction'}
        or type(gate['threshold']) not in {int, float}
        or not math.isfinite(gate['threshold'])
        or gate['threshold'] <= 0
    ):
        raise LabError('gate_schema')
    runs = {r['id']: r for r in case['runs']}
    if not all(isinstance(plan[k], str) and plan[k] in runs for k in ('candidate', 'reference')):
        raise LabError('unknown_run')
    observation = {r['time']: r['kw'] for r in case['observations']}
    series = []
    for key in ('candidate', 'reference'):
        run = runs[plan[key]]
        scale = 1000 if run['unit'] == 'MW' else 1
        series.append({r['time']: r['value'] * scale for r in run['rows']})
    common = sorted(set(observation) & set(series[0]) & set(series[1]))
    coverage = len(common) / len(case['expected_times'])
    allowed = (
        plan['operation'] == 'compare'
        and bool(common)
        and (len(common) if gate['basis'] == 'min_count' else coverage) >= gate['threshold']
    )
    errors = []
    for values in series:
        diffs = [abs(values[t] - observation[t]) for t in common]
        error = (
            (
                sum(diffs) / len(diffs)
                if plan['metric'] == 'mae'
                else math.sqrt(sum(x * x for x in diffs) / len(diffs))
            )
            if allowed
            else None
        )
        errors.append(round(error, 8) if error is not None else None)
    winner = (
        None
        if not allowed
        else (
            'tie'
            if errors[0] == errors[1]
            else plan['candidate' if errors[0] < errors[1] else 'reference']
        )
    )
    return {
        'status': 'compared' if allowed else 'insufficient_coverage',
        'times': common,
        'coverage': coverage,
        'candidate_error': errors[0],
        'reference_error': errors[1],
        'winner': winner,
    }


def oracle(case):
    """不调用计划执行器：遍历预期时刻，以独立连接/聚合路径给出参考结果。"""
    validate_case(case)
    request = case['request']
    chosen = [
        next(r for r in case['runs'] if r['id'] == request[k]) for k in ('candidate', 'reference')
    ]
    samples = []
    for t in sorted(case['expected_times']):
        obs = [r['kw'] for r in case['observations'] if r['time'] == t]
        forecasts = [
            [
                r['value'] / (0.001 if run['unit'] == 'MW' else 1)
                for r in run['rows']
                if r['time'] == t
            ]
            for run in chosen
        ]
        if obs and all(forecasts):
            samples.append((t, abs(forecasts[0][0] - obs[0]), abs(forecasts[1][0] - obs[0])))
    n = len(samples)
    fraction = n / len(case['expected_times'])
    ok = n > 0 and (n if case['rule']['version'] == 'v1' else fraction) >= case['rule']['minimum']
    values = []
    for col in (1, 2):
        value = None
        if ok:
            value = (
                math.fsum(
                    row[col] if request['metric'] == 'mae' else row[col] ** 2 for row in samples
                )
                / n
            )
            if request['metric'] == 'rmse':
                value = value**0.5
        values.append(round(value, 8) if value is not None else None)
    winner = (
        None
        if not ok
        else (
            'tie'
            if values[0] == values[1]
            else request['candidate' if values[0] < values[1] else 'reference']
        )
    )
    return {
        'status': 'compared' if ok else 'insufficient_coverage',
        'times': [r[0] for r in samples],
        'coverage': fraction,
        'candidate_error': values[0],
        'reference_error': values[1],
        'winner': winner,
    }


def verify_plan(case, output):
    errors = []
    actual = None
    try:
        actual = execute_plan(case, output)
        required = reference_plan(case)
        # 不只比较碰巧相同的数值；被跳过的安全步骤也必须失败。
        for key in required:
            # 已经排除bool和非有限数；2与2.0在阈值字段是同一数值。
            if key == 'gate':
                same = (
                    output[key]['basis'] == required[key]['basis']
                    and output[key]['threshold'] == required[key]['threshold']
                )
            else:
                same = canonical(output[key]) == canonical(required[key])
            if not same:
                errors.append('wrong_' + key)
        if actual != oracle(case):
            errors.append('wrong_result')
    except (LabError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return {'passed': not errors, 'errors': errors, 'result': actual}


def make_task(
    case, *, task_id, sequence, family, split, factors=(), panel='protocol', object_id=None
):
    validate_case(case)
    return {
        'task_id': task_id,
        'sequence': sequence,
        'family': family,
        'split': split,
        'rule_version': 'wind_' + case['rule']['version'],
        'messages': messages(case),
        'expected': oracle(case),
        'evaluation': {
            'kind': 'wind-plan-v2',
            'case': case,
            'reference_plan': reference_plan(case),
            'object_id': object_id or task_id,
            'factors': list(factors),
            'panel': panel,
            'origin': 'synthetic_protocol_pilot',
        },
    }


def validate_business_task(task):
    e = task['evaluation']
    case = e['case']
    if (
        set(e) != {'kind', 'case', 'reference_plan', 'object_id', 'factors', 'panel', 'origin'}
        or e['kind'] != 'wind-plan-v2'
        or e['origin'] != 'synthetic_protocol_pilot'
        or not isinstance(e['object_id'], str)
        or not e['object_id']
        or not isinstance(e['factors'], list)
        or not set(e['factors']) <= set(FACTORS)
    ):
        raise LabError('未知业务评价合同')
    validate_case(case)
    if (
        task['messages'] != messages(case)
        or task['expected'] != oracle(case)
        or e['reference_plan'] != reference_plan(case)
        or task['rule_version'] != 'wind_' + case['rule']['version']
    ):
        raise LabError('公开数据/规则与独立评价未绑定')


def build_suite():
    """固定合成生成器，跨实体拆分；重复模板不声称独立企业任务族。"""
    suite = []
    for version in ('v1', 'v2'):
        groups = [
            ('train', [(f,) for f in FACTORS], 'practice'),
            ('dev', list(itertools.combinations(FACTORS, 2)), 'development'),
        ]
        if version == 'v2':
            groups += [
                ('holdout', list(itertools.combinations(FACTORS, 3)), 'novel_composition'),
                ('holdout', [(f,) for f in FACTORS], 'retained_skill'),
                ('holdout', [('missing_rows',)], 'changed_rule'),
                ('holdout', [('missing_rows',)], 'abstain'),
            ]
        for split, combos, panel in groups:
            for factors in combos:
                seq = len(suite) + 1
                obj = f'b2_case_{seq:03d}'
                times = [f'{obj}_t{i}' for i in range(6)]
                obs = [300 + seq * 7 + i * 53 for i in range(6)]
                a = f'run_{seq * 3}'
                b = f'run_{seq * 3 + 1}'
                runs = []
                offsets_pair = ([0, 0, 0, 0, 0, 60], [15] * 6)
                if seq % 3 == 0:
                    offsets_pair = offsets_pair[::-1]
                for name, offsets in zip((a, b), offsets_pair):
                    mixed = 'mixed_units' in factors and name == a
                    rows = [
                        {'time': t, 'value': (y + offsets[i]) / (1000 if mixed else 1)}
                        for i, (t, y) in enumerate(zip(times, obs))
                    ]
                    if 'missing_rows' in factors:
                        rows = rows[:4] if name == a else rows[1:]
                    if 'shuffled_rows' in factors:
                        rows = rows[::2] + rows[1::2]
                    if panel == 'abstain':
                        rows = rows[:1] if name == a else rows[-1:]
                    runs.append({'id': name, 'unit': 'MW' if mixed else 'kW', 'rows': rows})
                if 'distractor' in factors:
                    runs.insert(
                        0,
                        {
                            'id': f'run_{seq * 3 + 2}',
                            'unit': 'kW',
                            'rows': [{'time': t, 'value': y} for t, y in zip(times, obs)],
                        },
                    )
                case = {
                    'request': {
                        'candidate': a,
                        'reference': b,
                        'metric': 'mae' if seq % 2 else 'rmse',
                    },
                    'rule': {'version': version, 'minimum': 2 if version == 'v1' else 0.8},
                    'expected_times': times,
                    'observations': [{'time': t, 'kw': y} for t, y in zip(times, obs)],
                    'runs': runs,
                }
                suite.append(
                    make_task(
                        case,
                        task_id=obj,
                        sequence=seq,
                        family=f'{split}_{panel}_' + '_'.join(factors),
                        split=split,
                        factors=factors,
                        panel=panel,
                        object_id=obj,
                    )
                )
    return suite


def audit_suite(tasks):
    from .lab import validate_task

    owners = {}
    seen_ids = set()
    seen_seq = set()
    structures = {}
    counts = {}
    for task in tasks:
        validate_task(task)
        e = task['evaluation']
        split = task['split']
        if task['task_id'] in seen_ids or task['sequence'] in seen_seq:
            raise LabError('重复任务或序号')
        seen_ids.add(task['task_id'])
        seen_seq.add(task['sequence'])
        objects = (
            [e['object_id']] + [r['id'] for r in e['case']['runs']] + e['case']['expected_times']
        )
        for obj in objects:
            if obj in owners and owners[obj] != split:
                raise LabError('跨切分实体或数据对象泄漏')
            owners[obj] = split
        if e['panel'] in {'practice', 'development', 'novel_composition'}:
            key = tuple(sorted(e['factors']))
            if key in structures and structures[key] != split:
                raise LabError('组合外推面板结构重复')
            structures[key] = split
        counts[split] = counts.get(split, 0) + 1
    return {
        'origin': 'synthetic_protocol_pilot',
        'task_count': len(tasks),
        'split_counts': counts,
        'cross_split_object_overlap': 0,
        'digest': digest(tasks),
        'limits': '同一受限计划语言内的合成协议；保留技能面板有意匹配训练难度，不是独立企业/语义族泛化。',
    }
