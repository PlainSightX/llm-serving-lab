"""单卡、DDP与FSDP2的真实训练；有效batch、数据与检查点合同固定。"""

import argparse
import contextlib
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from serving_lab.distributed_study import (
    rank_batches,
    loss_multiplier,
    validate_topology,
    validate_checkpoint_contract,
    validate_rank_contracts,
    verify_checkpoint_files,
)
from serving_lab.execution import write_once
from serving_lab.lab import inspect_base, digest, file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(args):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
        get_model_state_dict,
        StateDictOptions,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers import AutoModelForCausalLM

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if args.mode == 'single' and world != 1:
        raise ValueError('单卡模式不能有多个rank')
    if args.mode != 'single' and world != 2:
        raise ValueError('本包分布式模式要求两个rank')
    out = Path(args.output).resolve() / f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=False)
    plan = read(Path(args.data) / 'plan.json')
    records = read(Path(args.data) / 'samples.json')
    if file_hash(Path(args.data) / 'samples.json') != plan['samples_sha256']:
        raise ValueError('训练输入变化')
    base = inspect_base(args.model)
    if digest(base['files']) != plan['model_artifact_digest']:
        raise ValueError('训练基座变化')
    phase = dict(plan[args.phase])
    if args.scaling == 'weak':
        if args.phase != 'performance':
            raise ValueError('弱扩展只用于独立性能矩阵')
        phase.update(plan['weak_scaling'])
    global_batch = (
        plan['global_batch']
        if args.scaling == 'strong'
        else plan['weak_scaling']['global_batch_per_gpu'] * world
    )
    length = (
        phase['sequence_length']
        if 'sequence_length' in phase
        else plan['performance']['sequence_length']
    )
    steps = phase.get('steps', phase.get('warmup_steps', 0) + phase.get('measured_steps', 0))
    cfg = plan['performance'] if args.phase == 'recovery' else phase
    if args.profile:
        if args.phase != 'performance' or args.scaling != 'strong':
            raise ValueError('剖析只用于独立强扩展诊断')
        steps = cfg['warmup_steps'] + 2
    contract = {
        'schema': 1,
        'plan_sha256': file_hash(Path(args.data) / 'plan.json'),
        'model_digest': plan['model_artifact_digest'],
        'mode': args.mode,
        'phase': args.phase,
        'world_size': world,
        'precision': cfg['precision'],
        'global_batch': global_batch,
        'scaling': args.scaling,
        'sequence_length': length,
        'optimizer': cfg['optimizer'],
        'profile_enabled': args.profile,
        'planned_steps': steps,
    }
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.random.default_generator.manual_seed(plan['seed'])
    torch.cuda.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    prop = torch.cuda.get_device_properties(local)
    gpu_uuid = getattr(prop, 'uuid', None)
    if gpu_uuid is None:
        import ctypes, uuid

        class Uuid(ctypes.Structure):
            _fields_ = [('bytes', ctypes.c_ubyte * 16)]

        cuda = ctypes.CDLL('libcuda.so.1')
        value = Uuid()
        cuda.cuDeviceGetUuid.argtypes = [ctypes.POINTER(Uuid), ctypes.c_int]
        if cuda.cuDeviceGetUuid(ctypes.byref(value), local) != 0:
            raise RuntimeError('无法取得实际GPU UUID')
        gpu_uuid = 'GPU-' + str(uuid.UUID(bytes=bytes(value.bytes)))
    node = os.environ.get('N4_NODE_NAME', socket.gethostname())
    if node != os.environ.get('N4_EXPECTED_NODE', node):
        raise ValueError('调度节点与预定角色不符')
    member = {
        'rank': rank,
        'local_rank': local,
        'instance_id': os.environ['N4_INSTANCE_ID'],
        'node_name': node,
        'gpu_uuid': str(gpu_uuid),
        'gpu_name': prop.name,
        'total_memory': prop.total_memory,
        'pid': os.getpid(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'pid_namespace': os.readlink('/proc/self/ns/pid'),
    }
    initialized = False
    profiler = None
    profile_active = False

    def region(name):
        return torch.profiler.record_function(name) if args.profile else contextlib.nullcontext()

    try:
        if world > 1:
            dist.init_process_group(
                'nccl', timeout=datetime.timedelta(seconds=120), device_id=device
            )
            initialized = True
            members = [None] * world
            dist.all_gather_object(members, member)
        else:
            members = [member]
        topology = validate_topology(members, args.topology)
        source_root = Path(__file__).resolve().parents[1]
        source_hashes = {}
        for f in [Path(__file__), source_root / 'src/serving_lab/distributed_study.py']:
            relative = f.relative_to(source_root)
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.read_bytes())
            source_hashes[relative.as_posix()] = file_hash(f)
        signature = {
            'contract': contract,
            'config': cfg,
            'samples_sha256': file_hash(Path(args.data) / 'samples.json'),
            'source_hashes': source_hashes,
            'torch': torch.__version__,
            'transformers': importlib.metadata.version('transformers'),
            'nccl': torch.cuda.nccl.version(),
            'fault_requested': args.fault,
            'resume_manifest_sha256': file_hash(Path(args.resume) / 'files.json')
            if args.resume
            else None,
        }
        if world > 1:
            signatures = [None] * world
            dist.all_gather_object(signatures, signature)
            validate_rank_contracts(signatures)
        else:
            signatures = [signature]
        write_once(out / 'rank-contracts.json', signatures)
        write_once(
            out / 'input.json',
            {
                'contract': contract,
                'topology': topology,
                'config': cfg,
                'source_hashes': source_hashes,
                'torch': torch.__version__,
                'transformers': importlib.metadata.version('transformers'),
                'nccl': torch.cuda.nccl.version(),
                'fault_requested': args.fault,
                'resume': args.resume,
                'process_group_timeout_seconds': 120,
                'tf32': False,
                'deterministic_algorithms': True,
                'fsdp_param_dtype': 'float32; autocast only',
                'gradient_sync': 'last microstep only',
            },
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            attn_implementation='sdpa',
        ).to(device)
        model.config.use_cache = False
        model.train()
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
            raise ValueError('本包要求dropout=0')
        parameters = sum(p.numel() for p in model.parameters())
        if args.mode == 'ddp':
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=25,
            )
        elif args.mode == 'fsdp2':
            mesh = init_device_mesh('cuda', (world,), mesh_dim_names=('data',))
            policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
            for layer in model.model.layers:
                fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            fully_shard(model, mesh=mesh, mp_policy=policy)
        if cfg['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg['lr'],
                betas=tuple(cfg['betas']),
                eps=cfg['eps'],
                weight_decay=cfg['weight_decay'],
                foreach=False,
                fused=False,
            )
        ids = [r['id'] for r in records]
        by_id = {r['id']: r for r in records}
        if len(ids) < steps * global_batch:
            raise ValueError('冻结训练序列不足')
        amp = cfg['precision'] != 'fp32'
        write_once(
            out / 'model-ready.json',
            {
                'unique_parameters': parameters,
                'parameter_dtype': 'float32',
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
            },
        )
        print(
            json.dumps(
                {'rank': rank, 'model_ready': True, 'parameters': parameters, 'mode': args.mode}
            ),
            flush=True,
        )
        next_step = 0
        rng_states = []

        class State:
            def state_dict(self):
                model_state, optim_state = get_state_dict(model, optimizer)
                return {
                    'model': model_state,
                    'optimizer': optim_state,
                    'next_step': torch.tensor(next_step, dtype=torch.int64),
                    'rng': {str(i): value for i, value in enumerate(rng_states)},
                }

            def load_state_dict(self, state):
                nonlocal next_step, rng_states
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=state['model'],
                    optim_state_dict=state['optimizer'],
                )
                next_step = int(state['next_step'].item())
                rng_states = [state['rng'][str(i)] for i in range(world)]
                torch.set_rng_state(rng_states[rank]['cpu'])
                torch.cuda.set_rng_state(rng_states[rank]['cuda'], device)

        state = State()

        def gather_rng():
            value = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
            if world == 1:
                return [value]
            result = [None] * world
            dist.all_gather_object(result, value)
            return result

        if args.resume:
            checkpoint = Path(args.resume)
            meta = read(checkpoint / 'contract.json')
            expected_step = validate_checkpoint_contract(
                meta, contract, total_steps=steps, global_batch=global_batch
            )
            # 清单由收集端在全部rank文件取齐后写入；未提交的检查点不能消费。
            committed = read(checkpoint / 'files.json')
            verify_checkpoint_files(checkpoint, committed)
            metadata = dcp.FileSystemReader(str(checkpoint)).read_metadata()
            referenced = {info.relative_path for info in metadata.storage_data.values()}
            verify_checkpoint_files(checkpoint, committed, referenced=referenced)
            rng_states = gather_rng()
            dcp.load({'training': state}, checkpoint_id=str(checkpoint))
            if next_step != expected_step:
                raise ValueError('检查点步号与提交回执不同')
            write_once(
                out / 'resume.json',
                {
                    'next_step': next_step,
                    'checkpoint_manifest_sha256': file_hash(checkpoint / 'files.json'),
                },
            )
        rows = []
        started = time.monotonic()
        if args.profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        for step in range(next_step, steps):
            samples = rank_batches(
                ids, step=step, global_batch=global_batch, world_size=world, rank=rank
            )
            full_ids = ids[step * global_batch : (step + 1) * global_batch]
            total_tokens = sum(min(length, len(by_id[i]['input_ids'])) - 1 for i in full_ids)
            multiplier = loss_multiplier(world, total_tokens)
            if world > 1:
                dist.barrier()
            torch.cuda.synchronize()
            if step == cfg.get('warmup_steps', 0):
                torch.cuda.reset_peak_memory_stats()
            if profiler is not None and step == cfg['warmup_steps']:
                profiler.start()
                profile_active = True
            begin = time.monotonic()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device, dtype=torch.float64)
            for micro, sample_id in enumerate(samples):
                item = by_id[sample_id]['input_ids'][:length]
                valid = len(item)
                with region('n4_input_transfer'):
                    tokens = torch.tensor(
                        [item + [plan['pad_token_id']] * (length - valid)],
                        device=device,
                        dtype=torch.long,
                    )
                    attention = torch.arange(length, device=device)[None, :] < valid
                    labels = tokens.clone()
                    labels[:, valid:] = -100
                synchronize = micro == len(samples) - 1
                if args.mode == 'fsdp2':
                    model.set_requires_gradient_sync(synchronize)
                guard = (
                    model.no_sync()
                    if args.mode == 'ddp' and not synchronize
                    else contextlib.nullcontext()
                )
                with guard:
                    with (
                        region('n4_forward'),
                        torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp),
                    ):
                        output = model(input_ids=tokens, attention_mask=attention, use_cache=False)
                        logits = output.logits
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                            labels[:, 1:].reshape(-1),
                            reduction='sum',
                        )
                    local_loss.add_(loss.detach().double())
                    with region('n4_backward'):
                        (loss * multiplier).backward()
                del output, logits, loss, tokens, attention, labels
            if args.phase == 'correctness' and step == steps - 1:
                gradients = {}
                original = model.module if args.mode == 'ddp' else model
                for name, param in original.named_parameters():
                    if param.grad is None:
                        continue
                    grad = (
                        param.grad.full_tensor()
                        if hasattr(param.grad, 'full_tensor')
                        else param.grad
                    )
                    if rank == 0:
                        gradients[name] = grad.detach().cpu()
                if rank == 0:
                    torch.save(gradients, out / 'final-gradients.pt')
                del gradients
            with region('n4_optimizer'):
                optimizer.step()
            end_event.record()
            end_event.synchronize()
            wall = time.monotonic() - begin
            with region('n4_reporting'):
                if world > 1:
                    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            global_loss = float(local_loss.item() / total_tokens)
            if not (global_loss == global_loss and abs(global_loss) < 1e6):
                raise ValueError('训练loss不有限')
            row = {
                'step': step + 1,
                'rank': rank,
                'sample_ids': samples,
                'global_sample_ids': full_ids,
                'global_valid_tokens': total_tokens,
                'loss_multiplier': multiplier,
                'loss': global_loss,
                'processed_tokens': global_batch * length,
                'wall_seconds': wall,
                'cuda_event_seconds': start_event.elapsed_time(end_event) / 1000,
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'measured': args.phase == 'performance'
                and not args.profile
                and step >= cfg['warmup_steps'],
                'profiled': profile_active,
            }
            with (out / 'steps.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            rows.append(row)
            next_step = step + 1
            if rank == 0:
                print(
                    json.dumps({'step': next_step, 'loss': global_loss, 'seconds': wall}),
                    flush=True,
                )
            if profile_active:
                profiler.step()
            if (
                args.phase == 'recovery'
                and next_step == phase['checkpoint_step']
                and not args.resume
            ):
                rng_states = gather_rng()
                checkpoint = Path(args.output) / f'checkpoint-{next_step:04}'
                present = torch.tensor(int(checkpoint.exists()), device=device)
                if world > 1:
                    dist.all_reduce(present, op=dist.ReduceOp.MAX)
                if present.item():
                    raise ValueError('不覆盖旧检查点')
                if world > 1:
                    dist.barrier()
                dcp.save({'training': state}, checkpoint_id=str(checkpoint))
                if rank == 0:
                    write_once(
                        checkpoint / 'contract.json',
                        {
                            'contract': contract,
                            'next_step': next_step,
                            'completed_samples': next_step * global_batch,
                        },
                    )
                write_once(
                    out / 'checkpoint.json',
                    {'directory': str(checkpoint), 'step': next_step, 'saved': True},
                )
            if (
                args.fault
                and args.phase == 'recovery'
                and next_step == phase['fault_after_step']
                and rank == 1
            ):
                write_once(
                    out / 'injected-fault.json',
                    {
                        'after_step': next_step,
                        'checkpoint_step': phase['checkpoint_step'],
                        'exit_code': 73,
                        'wall_time_ns': time.time_ns(),
                    },
                )
                os._exit(73)
        if profile_active:
            profiler.stop()
            profile_active = False
            profiler.export_chrome_trace(str(out / 'trace.json'))
            import gzip, shutil

            with (
                (out / 'trace.json').open('rb') as original,
                gzip.open(out / 'trace.json.gz', 'xb') as packed,
            ):
                shutil.copyfileobj(original, packed)
        finite = torch.tensor(1, device=device)
        for param in model.parameters():
            tensor = param.to_local() if hasattr(param, 'to_local') else param
            finite.mul_(torch.isfinite(tensor).all().to(torch.int64))
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise ValueError('最终参数包含非有限值')
        if args.phase != 'performance':
            full = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            if rank == 0:
                torch.save(full, out / 'final-state.pt')
        if world > 1:
            dist.barrier()
        summary = {
            'completed': True,
            'rank': rank,
            'mode': args.mode,
            'phase': args.phase,
            'steps_completed': next_step,
            'steps_this_run': len(rows),
            'resumed': bool(args.resume),
            'elapsed_seconds': time.monotonic() - started,
            'measured_steps': sum(r['measured'] for r in rows),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'model_parameters': parameters,
            'contract': contract,
            'topology': topology,
            'application_qualified': False,
        }
        write_once(out / 'summary.json', summary)
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {
                'type': type(error).__name__,
                'message': str(error),
                'rank': rank,
                'wall_time_ns': time.time_ns(),
            },
        )
        raise
    finally:
        try:
            if profile_active:
                profiler.stop()
        finally:
            if initialized and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('model', 'data', 'output'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--mode', choices=['single', 'ddp', 'fsdp2'], required=True)
    p.add_argument('--phase', choices=['correctness', 'performance', 'recovery'], required=True)
    p.add_argument('--topology', choices=['single_gpu', 'single_node', 'two_node'], required=True)
    p.add_argument('--fault', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--scaling', choices=['strong', 'weak'], default='strong')
    p.add_argument('--profile', action='store_true')
    run(p.parse_args())
