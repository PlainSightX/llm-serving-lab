"""NCCL进程退出0不能掩盖数据错误或错误带宽口径。"""

import pytest


def test_nccl_rows_recompute_bus_bandwidth_and_reject_errors():
    from serving_lab.distributed_study import parse_nccl_result

    text = '''# nccl-tests version 2.17.6 nccl-headers=22705 nccl-library=22705
rank 0 nranks 2
rank 1 nranks 2
via NET/Socket/0
134217728 16777216 float none -1 46279.6 2.90 1.45 0 45509.7 2.95 1.47 0
# Out of bounds values : 0 OK
'''
    row = parse_nccl_result(text, 134217728, 'all_gather_perf')
    assert row['out_of_place']['bus_gbps'] == 1.45
    with pytest.raises(ValueError):
        parse_nccl_result(text, 134217728, 'reduce_scatter_perf')
    with pytest.raises(ValueError):
        parse_nccl_result(text.replace('1.45 0', '1.45 1'), 134217728, 'all_gather_perf')
    with pytest.raises(ValueError):
        parse_nccl_result(text.replace('2.90 1.45', '2.90 2.90'), 134217728, 'all_gather_perf')
