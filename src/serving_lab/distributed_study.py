"""N4共享的批次、有效token归一化和真实拓扑合同，不依赖GPU包。"""


def rank_batches(sample_ids, *, step, global_batch, world_size, rank):
    if (
        type(step) is not int
        or step < 0
        or type(world_size) is not int
        or world_size < 1
        or type(global_batch) is not int
        or global_batch < world_size
        or global_batch % world_size
        or type(rank) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError('无效的有效batch/rank配置')
    begin = step * global_batch
    end = begin + global_batch
    if end > len(sample_ids):
        raise ValueError('冻结样本序列不足，不隐式循环或重采样')
    selected = sample_ids[begin:end]
    if len(set(selected)) != global_batch:
        raise ValueError('同一有效batch存在重复样本身份')
    return selected[rank::world_size]


def loss_multiplier(world_size, global_valid_tokens):
    if (
        type(world_size) is not int
        or world_size < 1
        or type(global_valid_tokens) is not int
        or global_valid_tokens < 1
    ):
        raise ValueError('需要正整数world size和全局有效token数')
    # DDP/FSDP的梯度平均会再除world_size；不能平均不同长度样本的局部mean。
    return world_size / global_valid_tokens


def validate_topology(members, kind):
    count = len(members)
    if count not in {1, 2} or sorted(m['rank'] for m in members) != list(range(count)):
        raise ValueError('本包仅接受一个或两个完整rank')
    if any(not m['instance_id'] or not m['gpu_uuid'] for m in members):
        raise ValueError('缺少硬件身份')
    if len({m['gpu_uuid'] for m in members}) != count:
        raise ValueError('rank未独占不同物理GPU')
    hosts = len({m['instance_id'] for m in members})
    if all('node_name' in m for m in members) and len({m['node_name'] for m in members}) != hosts:
        raise ValueError('实际节点名与实例身份映射冲突')
    valid = (
        kind == 'single_gpu'
        and count == 1
        or kind == 'single_node'
        and count == 2
        and hosts == 1
        or kind == 'two_node'
        and count == 2
        and hosts == 2
    )
    if not valid:
        raise ValueError('声明拓扑与实例/GPU身份不符')
    return {'kind': kind, 'world_size': count, 'hosts': hosts, 'members': members}


def validate_checkpoint_contract(checkpoint, expected, *, total_steps, global_batch):
    if checkpoint.get('contract') != expected:
        raise ValueError('检查点训练合同与当前运行不符')
    step = checkpoint.get('next_step')
    if type(step) is not int or not 0 < step < total_steps:
        raise ValueError('恢复步号不在未完成范围')
    if checkpoint.get('completed_samples') != step * global_batch:
        raise ValueError('已消费样本数与步号不符')
    return step


def validate_rank_contracts(values):
    if not values or any(value != values[0] for value in values):
        raise ValueError('各rank的实际训练输入/源码/恢复身份不同')
    return values[0]


def verify_checkpoint_files(root, manifest, *, referenced=()):
    from pathlib import Path
    from .lab import file_hash

    root = Path(root).resolve()
    expected = manifest.get('files', {})
    if manifest.get('complete') is not True or not {'.metadata', 'contract.json'} <= set(expected):
        raise ValueError('检查点尚未完整提交')
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('检查点不接受链接')
        if path.is_file() and path.name != 'files.json':
            actual[path.relative_to(root).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError('检查点清单未覆盖实际文件集合')
    for name, sha in expected.items():
        if root not in actual[name].resolve().parents or file_hash(actual[name]) != sha:
            raise ValueError('检查点文件身份不符')
    for name in referenced:
        path = (root / name).resolve()
        if root not in path.parents or name not in expected:
            raise ValueError('DCP引用了清单外的分片')
    return expected


def parse_nccl_result(text, size, kind):
    import math, re

    if kind not in {'all_reduce_perf', 'all_gather_perf', 'reduce_scatter_perf'}:
        raise ValueError('未知collective')
    if (
        'nccl-headers=22705 nccl-library=22705' not in text
        or 'rank 0 nranks 2' not in text
        or 'rank 1 nranks 2' not in text
        or 'via NET/Socket/' not in text
        or '# Out of bounds values : 0 OK' not in text
    ):
        raise ValueError('缺少固定版本/双rank/Socket/正确性证据')
    rows = [
        line.split() for line in text.splitlines() if re.match(r'^\s*\d+\s+\d+\s+float\s', line)
    ]
    if len(rows) != 1 or len(rows[0]) != 13 or int(rows[0][0]) != size:
        raise ValueError('消息大小或结果行不符')
    row = rows[0]
    factor = 1.0 if kind == 'all_reduce_perf' else 0.5
    if row[3] != ('none' if kind == 'all_gather_perf' else 'sum') or row[4] != '-1':
        raise ValueError('collective操作列不符')
    expected_count = size // 4 if kind == 'all_reduce_perf' else size // 8
    if int(row[1]) != expected_count:
        raise ValueError('参与元素数不符')
    result = {'bytes': size, 'kind': kind}
    for label, offset in [('out_of_place', 5), ('in_place', 9)]:
        microseconds, algbw, busbw = map(float, row[offset : offset + 3])
        if (
            int(row[offset + 3]) != 0
            or not all(math.isfinite(v) and v >= 0 for v in (microseconds, algbw, busbw))
            or microseconds <= 0
        ):
            raise ValueError('NCCL数据检查失败或计时无效')
        derived = size / microseconds / 1000
        if abs(algbw - derived) > 0.006 or abs(busbw - derived * factor) > 0.006:
            raise ValueError('带宽和实际消息/时间不符')
        result[label] = {
            'microseconds': microseconds,
            'algorithm_gbps': algbw,
            'bus_gbps': busbw,
            'errors': 0,
        }
    return result


def validate_training_steps(rows, contract, records, *, rank, start_step, warmup):
    import math

    expected = list(range(start_step + 1, contract['planned_steps'] + 1))
    if not rows or [r['step'] for r in rows] != expected:
        raise ValueError('训练步序不完整、重复或乱序')
    ids = [r['id'] for r in records]
    by_id = {r['id']: r for r in records}
    batch = contract['global_batch']
    world = contract['world_size']
    length = contract['sequence_length']
    for row in rows:
        step = row['step'] - 1
        global_ids = ids[step * batch : (step + 1) * batch]
        local_ids = rank_batches(ids, step=step, global_batch=batch, world_size=world, rank=rank)
        tokens = sum(min(length, len(by_id[i]['input_ids'])) - 1 for i in global_ids)
        if (
            row['rank'] != rank
            or row['sample_ids'] != local_ids
            or row['global_sample_ids'] != global_ids
            or row['global_valid_tokens'] != tokens
            or row['processed_tokens'] != batch * length
            or row['loss_multiplier'] != loss_multiplier(world, tokens)
        ):
            raise ValueError('实际样本、rank或全局token分母不符')
        measured = (
            contract['phase'] == 'performance'
            and not contract['profile_enabled']
            and step >= warmup
        )
        profiled = contract['profile_enabled'] and step >= warmup
        if row['measured'] != measured or row['profiled'] != profiled:
            raise ValueError('预热、正式或剖析范围混入')
        for key in [
            'loss',
            'wall_seconds',
            'cuda_event_seconds',
            'peak_allocated_bytes',
            'peak_reserved_bytes',
        ]:
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('训练数值或测量无效')
    return expected


def validate_training_contract(contract, config, plan, plan_sha256):
    mode = contract['mode']
    phase = contract['phase']
    scaling = contract['scaling']
    world = contract['world_size']
    profile = contract['profile_enabled']
    if (
        mode not in {'single', 'ddp', 'fsdp2'}
        or phase not in {'correctness', 'performance', 'recovery'}
        or scaling not in {'strong', 'weak'}
        or type(profile) is not bool
        or world != (1 if mode == 'single' else 2)
        or scaling == 'weak'
        and phase != 'performance'
        or profile
        and (phase != 'performance' or scaling != 'strong')
    ):
        raise ValueError('运行类型不属于冻结矩阵')
    phase_config = dict(plan[phase])
    if scaling == 'weak':
        phase_config.update(plan['weak_scaling'])
    effective = dict(plan['performance']) if phase == 'recovery' else phase_config
    length = phase_config.get('sequence_length', plan['performance']['sequence_length'])
    steps = phase_config.get(
        'steps', phase_config.get('warmup_steps', 0) + phase_config.get('measured_steps', 0)
    )
    if profile:
        steps = effective['warmup_steps'] + 2
    expected = {
        'schema': 1,
        'plan_sha256': plan_sha256,
        'model_digest': plan['model_artifact_digest'],
        'mode': mode,
        'phase': phase,
        'world_size': world,
        'precision': effective['precision'],
        'global_batch': plan['global_batch']
        if scaling == 'strong'
        else plan['weak_scaling']['global_batch_per_gpu'] * world,
        'scaling': scaling,
        'sequence_length': length,
        'optimizer': effective['optimizer'],
        'profile_enabled': profile,
        'planned_steps': steps,
    }
    if contract != expected or config != effective:
        raise ValueError('运行自报合同或配置不符合冻结计划')
    return effective
