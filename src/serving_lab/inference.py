"""受控本地HF执行：运行器只消费公开消息，验证器在执行完成后读取答案。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from .execution import (
    DEFAULT_GENERATION,
    begin_execution,
    fail_execution,
    finish_execution,
    save_input,
    verify_execution,
    resume_existing,
)
from .lab import Lab, LabError, file_hash
from . import lab as lab_module, execution as execution_module
from . import business as business_module, memory as memory_module

SOURCE_SNAPSHOT = {
    'source_inference.py': Path(__file__).read_bytes(),
    'source_execution.py': execution_module.SOURCE_BYTES,
    'source_lab.py': lab_module.SOURCE_BYTES,
    'source_business.py': business_module.SOURCE_BYTES,
    'source_memory.py': memory_module.SOURCE_BYTES,
}


def assert_source_stable():
    for name, data in SOURCE_SNAPSHOT.items():
        if Path(__file__).with_name(name.removeprefix('source_')).read_bytes() != data:
            raise LabError('运行期间源码已修改，停止并保留当前执行，不把新磁盘代码冒充已执行版本')


def run_tasks(
    workspace,
    model_id,
    prefix,
    *,
    device='cuda',
    max_new_tokens=128,
    max_input_tokens=1024,
    arm='frozen',
    resume=False,
    memory_policy='none',
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    from peft import PeftModel

    if device == 'cuda' and not torch.cuda.is_available():
        raise LabError('CUDA不可用，不自动回退CPU')
    if device == 'cuda' and torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise LabError('可用显存不足4GiB，停止资格运行')
    spec = {
        **DEFAULT_GENERATION,
        'device': device,
        'max_new_tokens': max_new_tokens,
        'max_input_tokens': max_input_tokens,
    }
    results = []
    model = None
    tokenizer = None
    try:
        with Lab(workspace) as lab:
            if model_id == 'active':
                model_id = lab.active()
                if model_id is None:
                    raise LabError('没有活动版本')
            registered = lab.model(model_id)
            task_ids = [r['id'] for r in lab.db.execute('SELECT id FROM tasks ORDER BY seq')]
            if not task_ids:
                raise LabError('没有任务')
            for task_id in task_ids:
                assert_source_stable()
                if resume:
                    existing = resume_existing(lab, task_id, model_id, spec, arm, memory_policy)
                    if existing:
                        results.append(existing)
                        print(json.dumps(existing, ensure_ascii=False), flush=True)
                        continue
                attempt_id = f'{prefix}_{task_id}'
                # 开始调用前提交；同一候选不能换prefix重跑首次失败。
                request = begin_execution(
                    lab,
                    attempt_id,
                    task_id,
                    model_id,
                    generation=spec,
                    arm=arm,
                    memory_policy=memory_policy,
                )
                directory = lab.root / 'executions' / attempt_id
                try:
                    sources = {}
                    for source_name, source_data in SOURCE_SNAPSHOT.items():
                        target = directory / source_name
                        with target.open('xb') as f:
                            f.write(source_data)
                            f.flush()
                            os.fsync(f.fileno())
                        sources[target.name] = file_hash(target)
                    if model is None:
                        set_seed(spec['seed'])
                        base = registered['base_identity']['path']
                        tokenizer = AutoTokenizer.from_pretrained(
                            base, local_files_only=True, trust_remote_code=False
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            base,
                            local_files_only=True,
                            trust_remote_code=False,
                            use_safetensors=True,
                            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                            attn_implementation='sdpa',
                        )
                        if registered.get('kind') != 'base':
                            model = PeftModel.from_pretrained(
                                model, registered['artifact']['path'], local_files_only=True
                            )
                        model = model.to(device).eval()
                    # 此处永远不接收task.expected或反馈，只有公开投影。
                    messages = memory_module.input_messages(request)
                    rendered = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                    if len(ids) > max_input_tokens:
                        raise LabError('输入超限，不静默截断')
                    software = {
                        'source_files': sources,
                        'versions': {
                            name: importlib.metadata.version(name)
                            for name in ('torch', 'transformers', 'peft')
                        },
                        'tokenizer_eos_token_id': tokenizer.eos_token_id,
                        'template': 'apply_chat_template; enable_thinking=false; add_generation_prompt=true',
                    }
                    save_input(lab, attempt_id, rendered, ids, software)
                    tensor = torch.tensor([ids], device=device)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=tensor,
                            attention_mask=torch.ones_like(tensor),
                            use_model_defaults=False,
                            generation_config=GenerationConfig(
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            ),
                        )
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    generated = output[0, len(ids) :].tolist()
                    raw = tokenizer.decode(generated, skip_special_tokens=True)
                    assert_source_stable()
                    response = finish_execution(
                        lab,
                        attempt_id,
                        raw,
                        generated,
                        ended_eos=bool(generated and generated[-1] == tokenizer.eos_token_id),
                        elapsed_seconds=elapsed,
                        runtime={
                            'device': device,
                            'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                            'loaded_model_id': model_id,
                            'loaded_artifact_digest': registered['artifact']['digest'],
                            'loaded_parameter_identity': registered['parameter_identity'],
                            'prompt_tokens': len(ids),
                            'completion_tokens': len(generated),
                        },
                    )
                    verified = verify_execution(lab, attempt_id)
                    results.append(
                        {
                            'attempt_id': attempt_id,
                            'state': 'completed',
                            'raw_text': raw,
                            'output_complete': response['output_complete'],
                            'json_valid': response['json_valid'],
                            'generation_seconds': elapsed,
                            'evidence_level': 'model_smoke',
                        }
                    )
                    if lab.task(task_id)['split'] == 'train' and verified['complete']:
                        lab.feedback(attempt_id)
                except Exception as exc:
                    fail_execution(lab, attempt_id, exc)
                    results.append(
                        {
                            'attempt_id': attempt_id,
                            'state': 'failed',
                            'error': f'{type(exc).__name__}: {exc}',
                        }
                    )
                    # 加载/设备错误停止整轮；原预约和失败不删除。
                    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
                    raise
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            return {
                'model_id': model_id,
                'evidence_level': 'model_smoke',
                'executions': results,
                'new_model_calls': sum(not r.get('reused', False) for r in results),
                'limits': '小模型执行与门控资格；不证明持续学习或业务泛化。',
            }
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='受控真实模型任务执行，首次失败不可自动重跑')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--max-input-tokens', type=int, default=1024)
    parser.add_argument('--arm', default='frozen')
    parser.add_argument('--memory-policy', choices=sorted(memory_module.POLICIES), default='none')
    parser.add_argument(
        '--resume', action='store_true', help='只执行没有预约的后续任务；首次失败/未知保留'
    )
    args = parser.parse_args(argv)
    try:
        result = run_tasks(
            args.workspace,
            args.model,
            args.prefix,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            arm=args.arm,
            resume=args.resume,
            memory_policy=args.memory_policy,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LabError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(
            json.dumps({'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
