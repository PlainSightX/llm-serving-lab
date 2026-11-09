"""在目标Linux主机采集只读前检，结果供租赁后首次连接使用。"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess


def command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect(role, disk_path):
    release = {}
    path = Path('/etc/os-release')
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line:
                name, value = line.split('=', 1)
                release[name] = value.strip('"')
    memory = 0
    if Path('/proc/meminfo').exists():
        row = next(
            line
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemTotal:')
        )
        memory = int(row.split()[1]) * 1024
    cpu = os.cpu_count() or 0
    disk = shutil.disk_usage(disk_path).free
    virtual = command(['systemd-detect-virt'])
    pid1 = Path('/proc/1/comm').read_text().strip() if Path('/proc/1/comm').exists() else None
    gpu = []
    raw = command(
        [
            'nvidia-smi',
            '--query-gpu=name,uuid,memory.total,driver_version,compute_cap',
            '--format=csv,noheader,nounits',
        ]
    )
    if raw:
        for row in csv.reader(io.StringIO(raw)):
            name, identity, size, driver, capability = [value.strip() for value in row]
            gpu.append(
                {
                    'name': name,
                    'uuid': identity,
                    'memory_mib': int(size),
                    'driver': driver,
                    'compute_capability': capability,
                }
            )
    checks = {
        'linux_amd64': platform.system() == 'Linux' and platform.machine() == 'x86_64',
        'ubuntu_24_04': release.get('ID') == 'ubuntu' and release.get('VERSION_ID') == '24.04',
        'full_host_systemd': pid1 == 'systemd'
        and virtual not in {'docker', 'podman', 'lxc', 'wsl'}
        and not Path('/.dockerenv').exists(),
        'cpu': cpu >= (8 if role == 'gpu' else 4),
        'memory': memory >= (30 if role == 'gpu' else 7) * 1024**3,
        'free_disk': disk >= (150 if role == 'gpu' else 30) * 1024**3,
    }
    if role == 'gpu':
        checks['gpu_target'] = len(gpu) >= 1 and all(
            item['memory_mib'] >= 22000
            and float(item['compute_capability']) >= 8
            and int(item['driver'].split('.')[0]) >= 580
            for item in gpu
        )
        checks['nvidia_container_runtime'] = shutil.which('nvidia-container-runtime') is not None
    if role == 'loadgen':
        checks['docker_cli'] = shutil.which('docker') is not None
        checks['docker_daemon'] = (
            checks['docker_cli']
            and command(
                [
                    'docker',
                    '--host',
                    'unix:///var/run/docker.sock',
                    'info',
                    '--format',
                    '{{.ServerVersion}}',
                ]
            )
            is not None
        )
    return {
        'role': role,
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'os_release': release,
        'virtualization': virtual,
        'pid1': pid1,
        'cpu_count': cpu,
        'memory_bytes': memory,
        'disk_free_bytes': disk,
        'gpus': gpu,
        'checks': checks,
        'passed': all(checks.values()),
        'scope': 'read-only host measurements against the project rental target',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['gpu', 'control', 'loadgen'], required=True)
    parser.add_argument('--disk-path', default='/')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('前检输出必须为新文件')
    result = collect(args.role, args.disk_path)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'checks': result['checks']}))
    raise SystemExit(0 if result['passed'] else 2)
