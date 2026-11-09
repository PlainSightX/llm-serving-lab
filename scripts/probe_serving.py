"""真实引擎直连资格：公开请求、实际流片段、usage、token计数和释放事件。"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import httpx

from serving_lab.engine import EngineBinding, ReleaseJournal
from serving_lab.execution import write_once


def probe(binding_path, output_dir):
    binding = EngineBinding.load(binding_path)
    if not binding.alive():
        raise RuntimeError('启动代际不匹配')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = ReleaseJournal(
        binding.events_path, binding.worker_instance_id, binding.startup_generation
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        binding.model_path, local_files_only=True, trust_remote_code=False
    )
    cases = [('plain', '请只输出：OK', False), ('stream', '请用一句中文说明什么是风力发电。', True)]
    summary = []
    with httpx.Client(base_url=binding.endpoint, timeout=60, trust_env=False) as client:
        (output / 'metrics-before.txt').write_text(client.get('/metrics').text, encoding='utf-8')
        for label, prompt, stream in cases:
            request_id = uuid.uuid4().hex
            messages = [{'role': 'user', 'content': prompt}]
            body = {
                'model': binding.served_model,
                'messages': messages,
                'max_tokens': 128,
                'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False},
                'stream': stream,
            }
            if stream:
                body['stream_options'] = {'include_usage': True}
            local_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            tokenized = client.post(
                '/tokenize',
                json={
                    'model': binding.served_model,
                    'messages': messages,
                    'add_generation_prompt': True,
                    'chat_template_kwargs': {'enable_thinking': False},
                },
            )
            tokenized.raise_for_status()
            write_once(
                output / (label + '-request.json'),
                {
                    'request_id': request_id,
                    'body': body,
                    'binding_sha256': binding.manifest_sha256,
                    'local_input_ids': local_tokens,
                    'engine_tokenize': tokenized.json(),
                },
            )
            started = time.perf_counter()
            first = None
            events = []
            content = ''
            usage = None
            done = False
            finish = None
            if stream:
                with client.stream(
                    'POST', '/v1/chat/completions', json=body, headers={'X-Request-Id': request_id}
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        events.append({'elapsed': time.perf_counter() - started, 'data': data})
                        if data == '[DONE]':
                            done = True
                            break
                        event = json.loads(data)
                        if event.get('usage'):
                            usage = event['usage']
                        for choice in event.get('choices', []):
                            text = choice.get('delta', {}).get('content') or ''
                            if text and first is None:
                                first = time.perf_counter() - started
                            content += text
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
            else:
                response = client.post(
                    '/v1/chat/completions', json=body, headers={'X-Request-Id': request_id}
                )
                response.raise_for_status()
                event = response.json()
                events = [event]
                content = event['choices'][0]['message']['content']
                usage = event['usage']
                finish = event['choices'][0]['finish_reason']
                done = True
            elapsed = time.perf_counter() - started
            release = None
            for _ in range(100):
                release = journal.get(request_id)
                if release:
                    break
                time.sleep(0.05)
            result = {
                'label': label,
                'request_id': request_id,
                'content': content,
                'events': events,
                'usage': usage,
                'finish_reason': finish,
                'done': done,
                'ttft_seconds': first,
                'elapsed_seconds': elapsed,
                'release': release,
                'token_counts_match': tokenized.json()['count']
                == len(local_tokens)
                == (usage or {}).get('prompt_tokens'),
                'evidence': 'real_local_vllm_qualification_not_research_or_speedup',
            }
            write_once(output / (label + '-response.json'), result)
            if not (done and content and release and result['token_counts_match']):
                raise RuntimeError('真实响应/释放/token合同未通过，保留原输出')
            summary.append(
                {
                    k: result[k]
                    for k in (
                        'label',
                        'request_id',
                        'usage',
                        'elapsed_seconds',
                        'ttft_seconds',
                        'token_counts_match',
                    )
                }
            )
        (output / 'metrics-after.txt').write_text(client.get('/metrics').text, encoding='utf-8')
    write_once(
        output / 'summary.json',
        {
            'passed': True,
            'calls': len(cases),
            'results': summary,
            'worker_instance_id': binding.worker_instance_id,
        },
    )
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.binding, args.output), ensure_ascii=False, indent=2))
