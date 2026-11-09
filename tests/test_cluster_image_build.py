"""失败的构建上下文需要保留，后续调用应明确拒绝消费半成品。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_partial_context_is_rejected_before_docker_build(tmp_path):
    from build_cluster_images import run

    (tmp_path / 'context').mkdir()
    retained = tmp_path / 'context/partial'
    retained.write_text('original failure evidence')
    with pytest.raises(ValueError, match='尚未冻结'):
        run(tmp_path, 'control')
    assert retained.read_text() == 'original failure evidence'
