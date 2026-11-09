"""本机 Linux 运行根：推理环境、编译工具链与原生运行目录都放在这里，不放进仓库。

默认 ~/.local/share/llm-serving-lab，可用环境变量 SERVING_LAB_ROOT 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path


def lab_root():
    return Path(os.environ.get('SERVING_LAB_ROOT') or Path.home() / '.local/share/llm-serving-lab')


def expand(path):
    """配置里的路径允许写 ~ 与 $HOME 等环境变量。"""
    return Path(os.path.expandvars(os.path.expanduser(str(path))))
