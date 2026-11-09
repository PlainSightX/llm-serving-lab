"""分布式批次和损失口径不能因rank拆分改变。"""

import pytest


def test_global_sample_plan_is_preserved_across_one_and_two_ranks():
    from serving_lab.distributed_study import rank_batches

    one = rank_batches(list(range(12)), step=1, global_batch=4, world_size=1, rank=0)
    left = rank_batches(list(range(12)), step=1, global_batch=4, world_size=2, rank=0)
    right = rank_batches(list(range(12)), step=1, global_batch=4, world_size=2, rank=1)
    assert one == [4, 5, 6, 7] and left == [4, 6] and right == [5, 7]
    assert sorted(left + right) == one
    with pytest.raises(ValueError):
        rank_batches(list(range(12)), step=0, global_batch=3, world_size=2, rank=0)


def test_rank_averaging_recovers_global_token_weighted_loss():
    from serving_lab.distributed_study import loss_multiplier

    sums = [3.0, 10.0]
    tokens = [1, 5]
    distributed = sum(s * loss_multiplier(2, sum(tokens)) for s in sums) / 2
    assert distributed == pytest.approx(sum(sums) / sum(tokens))
    assert distributed != pytest.approx(sum(s / t for s, t in zip(sums, tokens)) / 2)
    with pytest.raises(ValueError):
        loss_multiplier(2, 0)


def test_topology_cannot_claim_two_gpus_or_one_host_from_wrong_identities():
    from serving_lab.distributed_study import validate_topology

    members = [
        {'rank': 0, 'instance_id': 'host-a', 'gpu_uuid': 'gpu-a'},
        {'rank': 1, 'instance_id': 'host-b', 'gpu_uuid': 'gpu-b'},
    ]
    assert validate_topology(members, 'two_node')['world_size'] == 2
    with pytest.raises(ValueError):
        validate_topology(members, 'single_node')
    duplicate = [members[0], {**members[1], 'gpu_uuid': 'gpu-a'}]
    with pytest.raises(ValueError):
        validate_topology(duplicate, 'two_node')
    assert (
        validate_topology([members[0], {**members[1], 'instance_id': 'host-a'}], 'single_node')[
            'hosts'
        ]
        == 1
    )


def test_checkpoint_cannot_resume_with_changed_training_contract():
    from serving_lab.distributed_study import validate_checkpoint_contract

    contract = {
        'plan_sha256': 'a' * 64,
        'model_digest': 'b' * 64,
        'mode': 'fsdp2',
        'phase': 'recovery',
        'world_size': 2,
    }
    checkpoint = {'contract': contract, 'next_step': 4, 'completed_samples': 16}
    assert validate_checkpoint_contract(checkpoint, contract, total_steps=12, global_batch=4) == 4
    with pytest.raises(ValueError):
        validate_checkpoint_contract(
            checkpoint, {**contract, 'world_size': 1}, total_steps=12, global_batch=4
        )
    with pytest.raises(ValueError):
        validate_checkpoint_contract(
            {**checkpoint, 'completed_samples': 20}, contract, total_steps=12, global_batch=4
        )


def test_rank_contract_handshake_rejects_locally_valid_but_different_sources():
    from serving_lab.distributed_study import validate_rank_contracts

    source = {
        'contract': {'world_size': 2},
        'samples_sha256': 'a',
        'source_hashes': {'trainer': 'b'},
        'resume_manifest_sha256': None,
    }
    assert validate_rank_contracts([source, source]) == source
    with pytest.raises(ValueError):
        validate_rank_contracts([source, {**source, 'samples_sha256': 'other'}])
    with pytest.raises(ValueError):
        validate_rank_contracts([source, {**source, 'resume_manifest_sha256': 'other'}])


def test_checkpoint_manifest_must_cover_every_actual_file_and_dcp_reference(tmp_path):
    from serving_lab.distributed_study import verify_checkpoint_files
    from serving_lab.lab import file_hash

    for name in ['.metadata', 'contract.json', '__0_0.distcp']:
        (tmp_path / name).write_bytes(b'valid')
    manifest = {'complete': True, 'files': {p.name: file_hash(p) for p in tmp_path.iterdir()}}
    verify_checkpoint_files(tmp_path, manifest, referenced=['__0_0.distcp'])
    (tmp_path / '__1_0.distcp').write_bytes(b'unlisted')
    with pytest.raises(ValueError):
        verify_checkpoint_files(tmp_path, manifest, referenced=['__1_0.distcp'])
    with pytest.raises(ValueError):
        verify_checkpoint_files(tmp_path, manifest, referenced=['../outside.distcp'])
