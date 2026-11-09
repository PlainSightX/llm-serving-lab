"""同一open-loop计量器测直连和网关；预定到达、失败、实际长度均留证。"""

from __future__ import annotations
import asyncio
import json
import math
import time
from collections import Counter

import httpx


class StreamObservation:
    def __init__(self, model, prompt_tokens, budget, allowed_models=None):
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.budget = budget
        self.allowed_models = set(allowed_models or [model])
        self.actual_model = None
        self.first_content = None
        self.last_content = None
        self.done = None
        self.finish = None
        self.content = ''
        self.usage = None
        self.error = None
        self.events = []

    def feed(self, data, elapsed):
        self.events.append({'elapsed': elapsed, 'data': data})
        if data == '[DONE]':
            self.done = elapsed
            return
        value = json.loads(data)
        if 'error' in value:
            self.error = value['error']
            return
        model = value.get('model')
        if (
            model not in self.allowed_models
            or self.actual_model is not None
            and model != self.actual_model
        ):
            self.error = 'unexpected_model'
        if self.actual_model is None:
            self.actual_model = model
        if value.get('usage'):
            self.usage = value['usage']
        for choice in value.get('choices', []):
            content = choice.get('delta', {}).get('content') or ''
            if content:
                if self.first_content is None:
                    self.first_content = elapsed
                self.last_content = elapsed
                self.content += content
            if choice.get('finish_reason'):
                self.finish = choice['finish_reason']

    @property
    def valid(self):
        u = self.usage or {}
        return bool(
            not self.error
            and self.done is not None
            and self.first_content is not None
            and self.finish in {'stop', 'length'}
            and u.get('prompt_tokens') == self.prompt_tokens
            and 0 < u.get('completion_tokens', 0) <= self.budget
            and u.get('total_tokens') == u.get('prompt_tokens', 0) + u.get('completion_tokens', 0)
            and (self.finish != 'length' or u['completion_tokens'] == self.budget)
        )


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    return {
        'count': len(values),
        'median': percentile(values, 0.5),
        'p95_descriptive': percentile(values, 0.95),
        'max': max(values) if values else None,
    }


def summarize(rows, duration, slo, *, interval, lag_fraction=0.1):
    valid = [r for r in rows if r['protocol_valid']]
    good = [
        r
        for r in valid
        if r['ttft_scheduled'] <= slo['ttft_seconds']
        and r['e2e_scheduled']
        <= (slo['short_e2e_seconds'] if r['output_budget'] == 16 else slo['long_e2e_seconds'])
    ]
    lag = percentile([r['send_lag'] for r in rows], 0.95)
    return {
        'planned': len(rows),
        'sent': sum(r['sent'] for r in rows),
        'protocol_valid': len(valid),
        'slo_met': len(good),
        'slo_fraction': len(good) / len(rows),
        'slo_goodput_rps': len(good) / duration,
        'elapsed_seconds': duration,
        'completed_output_tps': sum(r['usage']['completion_tokens'] for r in valid) / duration,
        'status_counts': dict(Counter(str(r['status']) for r in rows)),
        'finish_counts': dict(Counter(r.get('finish_reason') for r in valid)),
        'send_lag_p95': lag,
        'schedule_valid': all(r['sent'] for r in rows) and lag <= interval * lag_fraction,
        'ttft_scheduled': distribution([r['ttft_scheduled'] for r in valid]),
        'e2e_scheduled': distribution([r['e2e_scheduled'] for r in valid]),
        'ttft_sent': distribution([r['ttft_sent'] for r in valid]),
        'output_tokens': distribution([r['usage']['completion_tokens'] for r in valid]),
    }


async def window(endpoint, requests, offsets, config, *, action=None):
    """无发送semaphore；日志在窗口完成后由调用者写盘，迟发不补成突发。"""
    rows = [None] * len(requests)
    loop = asyncio.get_running_loop()
    from .tls import client_context

    verify = client_context(config['tls']) if config.get('tls') else True
    async with httpx.AsyncClient(
        base_url=endpoint,
        timeout=config['client_timeout_seconds'],
        trust_env=False,
        verify=verify,
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
    ) as client:
        started = loop.time() + 0.1

        async def one(i, item):
            scheduled = started + offsets[i]
            await asyncio.sleep(max(0, scheduled - loop.time()))
            sent_at = loop.time()
            lag = sent_at - scheduled
            row = {
                k: item[k]
                for k in ('id', 'shape', 'expected_version', 'input_tokens', 'output_budget')
            }
            row.update(
                scheduled_offset=offsets[i],
                send_offset=sent_at - started,
                send_lag=lag,
                sent=False,
                status=None,
                protocol_valid=False,
            )
            rows[i] = row
            if lag > config['late_drop_seconds']:
                row['error'] = 'client_schedule_missed'
                return
            row['sent'] = True
            observation = StreamObservation(
                item['expected_version'],
                item['input_tokens'],
                item['output_budget'],
                item.get('allowed_versions'),
            )
            try:
                target = item.get('endpoint', '').rstrip('/') + '/v1/chat/completions'
                async with client.stream(
                    'POST', target, json=item['body'], headers=item['headers']
                ) as response:
                    row.update(
                        status=response.status_code,
                        response_headers=dict(response.headers),
                        headers_elapsed=loop.time() - sent_at,
                    )
                    if response.status_code != 200:
                        row['response_error'] = (await response.aread()).decode()
                        return
                    if item.get('path') == 'gateway':
                        actual = response.headers.get('x-model-version')
                        allowed = item.get('allowed_versions') or [item['expected_version']]
                        profile = item.get('profiles', {}).get(actual, item['expected_profile'])
                        if (
                            actual not in allowed
                            or response.headers.get('x-serving-profile') != profile
                        ):
                            row['error'] = 'gateway_identity_mismatch'
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            observation.feed(line[5:].strip(), loop.time() - sent_at)
                            if observation.done is not None:
                                break
            except Exception as exc:
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            finally:
                row.update(
                    protocol_valid=observation.valid and not row.get('error'),
                    events=observation.events,
                    finish_reason=observation.finish,
                    usage=observation.usage,
                    content=observation.content,
                    actual_version=observation.actual_model,
                    planned_version_match=observation.actual_model == item['expected_version'],
                    ttft_sent=observation.first_content,
                    ttft_scheduled=lag + observation.first_content
                    if observation.first_content is not None
                    else None,
                    e2e_sent=observation.done,
                    e2e_scheduled=lag + observation.done if observation.done is not None else None,
                    end_offset=loop.time() - started,
                    protocol_error=observation.error,
                )
                if (
                    item.get('path') == 'gateway'
                    and observation.actual_model is not None
                    and row['response_headers'].get('x-model-version') != observation.actual_model
                ):
                    row.update(protocol_valid=False, error='header_and_stream_model_mismatch')

        tasks = [asyncio.create_task(one(i, r)) for i, r in enumerate(requests)]
        action_task = asyncio.create_task(action(started)) if action else None
        await asyncio.gather(*tasks)
        try:
            action_result = await action_task if action_task else None
        except Exception as exc:
            action_result = {'error': {'type': type(exc).__name__, 'message': str(exc)}}
        elapsed = max(loop.time() - started, offsets[-1] + (offsets[1] - offsets[0]))
    return {
        'rows': rows,
        'elapsed_seconds': elapsed,
        'action': action_result,
        'started_monotonic': started,
        'ended_monotonic': loop.time(),
    }


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith('vllm:'):
            key, value = line.rsplit(' ', 1)
            try:
                result[key] = float(value)
            except ValueError:
                pass
    return result


def metric_delta(before, after, name):
    keys = {k for k in after if k.split('{')[0] == name}
    if keys != {k for k in before if k.split('{')[0] == name}:
        return None
    if not keys or any(
        not math.isfinite(after[k]) or not math.isfinite(before[k]) or after[k] < before[k]
        for k in keys
    ):
        return None
    return sum(after[k] - before[k] for k in keys)


def metric_series_delta(before, after, name):
    result = {}
    for key in sorted(set(before) | set(after)):
        if key.split('{')[0] != name:
            continue
        valid = (
            key in before
            and key in after
            and math.isfinite(before[key])
            and math.isfinite(after[key])
            and after[key] >= before[key]
        )
        a = before.get(key)
        b = after.get(key)
        result[key] = {
            'before': a if a is not None and math.isfinite(a) else None,
            'after': b if b is not None and math.isfinite(b) else None,
            'delta': b - a if valid else None,
            'valid': valid,
        }
    return result
