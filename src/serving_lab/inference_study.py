"""N2的固定质量口径；负载长度测试与自然停止的质量评价分开。"""

from decimal import Decimal, InvalidOperation
import re

# 本包在任何候选校准前冻结；更换面板应使用新版本，而不是沿用资格名。
QUALITY_PLAN_SHA256 = '04f928549b34a3bbed82fa1da05e17f9ebf907ea5441b2341dc1fba033142660'

QUALITY_INSTRUCTION = (
    'Solve the math problem carefully. Show concise reasoning. '
    'Finish your response with one final line exactly in the form: #### <numeric answer>. '
    'Do not put currency symbols, units, or any text after that number.'
)


def quality_messages(row):
    return [
        {'role': 'system', 'content': QUALITY_INSTRUCTION},
        {'role': 'user', 'content': row['question']},
    ]


def numeric_answer(text):
    matches = re.findall(r'(?:^|\n)####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\Z', text.strip())
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(',', ''))
    except InvalidOperation:
        return None


def quality_pass(content, reference, finish_reason):
    expected = numeric_answer(reference)
    if expected is None:
        raise ValueError('官方答案缺少有效数值终行')
    return finish_reason == 'stop' and numeric_answer(content) == expected


def require_variant(manifest, variant):
    """候选角色由实际服务profile核对，不能靠CLI标签逃过量化门。"""
    profile = manifest['profile']
    quant = profile.get('quantization')
    if profile.get('dtype') not in {'bfloat16', 'bf16'}:
        raise ValueError('本包尚未取得其他dtype的桥接基线')
    if variant == 'bf16' and quant is None:
        return
    if (
        variant == 'awq'
        and isinstance(quant, dict)
        and quant.get('quant_method') == 'compressed-tensors'
    ):
        groups = quant.get('config_groups', {})
        if groups and all(
            g.get('weights', {}).get('num_bits') == 4 and g['weights'].get('group_size') == 128
            for g in groups.values()
        ):
            return
    raise ValueError('声明variant与实际量化profile不符')


def require_full_quality(proof, *, require_pass=True):
    if (
        proof.get('raw_verified') is not True
        or type(proof.get('gate_passed')) is not bool
        or require_pass
        and proof['gate_passed'] is not True
        or proof.get('smoke') is not False
        or proof.get('cases') != 256
        or proof.get('plan_sha256') != QUALITY_PLAN_SHA256
        or type(proof.get('correct')) is not int
        or not 0 <= proof['correct'] <= 256
        or proof.get('score') != proof['correct'] / 256
        or require_pass
        and proof['score'] < 0.70
    ):
        raise ValueError('缺少完整256题的独立质量证明')


def require_quality_proof(proof, manifest, *, diagnostic_rejected=False):
    require_full_quality(proof, require_pass=not diagnostic_rejected)
    require_variant(manifest, proof.get('variant'))
    if diagnostic_rejected and (proof['gate_passed'] is not False or proof['variant'] != 'awq'):
        raise ValueError('拒绝候选诊断只接受已审计失败的AWQ')
    if (
        proof.get('artifact_digest') != manifest['artifact_digest']
        or proof.get('profile_digest') != manifest['profile_digest']
    ):
        raise ValueError('质量证明未绑定当前工件和profile')


def require_baseline(proof, plan_sha256):
    require_full_quality(proof)
    if proof.get('variant') != 'bf16' or proof.get('plan_sha256') != plan_sha256:
        raise ValueError('量化基线必须是同面板的已审计BF16')


def summarize_fixed(rows, duration, rate=None):
    """TPOT是逐请求平均输出token耗时，不把SSE块间隔称为逐token分布。"""
    from .benchmark import distribution, percentile
    from collections import Counter

    valid = [
        r
        for r in rows
        if r['protocol_valid']
        and r['finish_reason'] == 'length'
        and r['usage']['completion_tokens'] == r['output_budget']
    ]
    good = []
    tpot = []
    for row in valid:
        value = (row['e2e_sent'] - row['ttft_sent']) / (row['usage']['completion_tokens'] - 1)
        tpot.append(value)
        first_limit = 3 if row['shape'].startswith('prefill') else 1
        if (
            row['ttft_scheduled'] <= first_limit
            and value <= 0.08
            and row['e2e_scheduled'] <= 3 + 0.08 * row['output_budget']
        ):
            good.append(row)
    lag = percentile([r['send_lag'] for r in rows], 0.95)
    schedule = all(r['sent'] for r in rows) and (rate is None or lag <= 0.1 / rate)
    return {
        'planned': len(rows),
        'sent': sum(r['sent'] for r in rows),
        'fixed_length_valid': len(valid),
        'status_counts': dict(Counter(str(r['status']) for r in rows)),
        'slo_met': len(good),
        'slo_fraction': len(good) / len(rows),
        'schedule_valid': schedule,
        'target_met': schedule and len(good) / len(rows) >= 0.99,
        'duration_seconds': duration,
        'completed_rps': len(valid) / duration,
        'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in valid) / duration,
        'ttft': distribution([r['ttft_scheduled'] for r in valid]),
        'e2e': distribution([r['e2e_scheduled'] for r in valid]),
        'tpot': distribution(tpot),
        'send_lag_p95': lag,
    }
