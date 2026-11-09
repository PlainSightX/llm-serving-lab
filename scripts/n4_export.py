"""把已核恢复后的完整参数导出为既有服务可消费工件，不继承应用资格。"""

import argparse
import gc
import json
from pathlib import Path

from serving_lab.execution import write_once
from serving_lab.lab import file_hash, inspect_base, digest
from serving_lab.parameters import parameter_identity
from n4_compare import tensor_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    training = Path(args.training)
    out = Path(args.output)
    base = Path(args.base)
    data = Path(args.data)
    summary = read(training / 'summary.json')
    comparison = read(args.comparison)
    plan = read(data / 'plan.json')
    if not summary['completed'] or not summary['resumed'] or summary['phase'] != 'recovery':
        raise ValueError('必须使用已完成的真实恢复产物')
    if (
        not comparison['passed']
        or comparison['final_step'] != plan['recovery']['steps']
        or comparison['input_sha256']['candidate'] != file_hash(training / 'input.json')
        or comparison['summary_sha256']['candidate'] != file_hash(training / 'summary.json')
        or comparison['steps_sha256']['candidate'] != file_hash(training / 'steps.jsonl')
        or comparison['state_sha256']['candidate'] != file_hash(training / 'final-state.pt')
        or summary['contract']['plan_sha256'] != file_hash(data / 'plan.json')
    ):
        raise ValueError('数值证据未绑定该产物')
    if digest(inspect_base(base)['files']) != plan['model_artifact_digest']:
        raise ValueError('父参数身份不符')
    out.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation='sdpa',
    ).eval()
    state = torch.load(training / 'final-state.pt', map_location='cpu', weights_only=True)
    if comparison['state_sha256']['candidate'] != file_hash(training / 'final-state.pt'):
        raise ValueError('读取期间最终参数变化')
    update = tensor_metrics(model.state_dict(), state)
    if update['changed_tensors'] == 0 or update['relative_l2'] == 0:
        raise ValueError('未观察到真实数值更新')
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    sample = read(data / 'samples.json')[0]['input_ids'][:64]
    probe = torch.tensor([sample])
    with torch.inference_mode():
        before = model(probe, use_cache=False).logits[:, -1].clone()
    model.save_pretrained(out / 'model', safe_serialization=True)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(out / 'model')
    write_once(
        out / 'model/source.json',
        {
            'kind': 'n4_fsdp2_recovered_full_parameter_candidate',
            'parent_artifact_digest': plan['model_artifact_digest'],
            'plan_sha256': file_hash(data / 'plan.json'),
            'training_summary_sha256': file_hash(training / 'summary.json'),
            'final_state_sha256': file_hash(training / 'final-state.pt'),
            'comparison_sha256': file_hash(args.comparison),
            'application_qualified': False,
        },
    )
    reference = model.state_dict()
    del model
    gc.collect()
    reloaded = AutoModelForCausalLM.from_pretrained(
        out / 'model',
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation='sdpa',
    ).eval()
    equality = tensor_metrics(reference, reloaded.state_dict())
    del reference
    with torch.inference_mode():
        after = reloaded(probe, use_cache=False).logits[:, -1]
    if equality['max_abs'] != 0 or not torch.equal(before, after):
        raise ValueError('工件保存重载不一致')
    parameters = parameter_identity(out / 'model')
    identity = inspect_base(out / 'model')
    result = {
        'completed': True,
        'parent_numeric_delta': update,
        'reload_parameters': equality,
        'reload_logit_max_abs': float((before - after).abs().max()),
        'artifact_digest': digest(identity['files']),
        'parameter_digest': parameters['digest'],
        'tensor_count': parameters['tensor_count'],
        'unique_parameters': sum(p.numel() for p in reloaded.parameters()),
        'storage_dtype': 'float32',
        'model_identity': identity,
        'training': str(training),
        'source_sha256': file_hash(__file__),
        'application_qualified': False,
        'publication_status': 'not_published',
    }
    write_once(out / 'receipt.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['training', 'output', 'base', 'data', 'comparison']:
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
