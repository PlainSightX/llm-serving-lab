"""仅下载并解包固定官方包到项目私有目录；不执行系统安装或维护脚本。"""

import ctypes
import hashlib
import json
import os
import shlex
import subprocess
import sysconfig
from pathlib import Path

base = Path(
    os.environ.get('SERVING_LAB_ROOT') or Path.home() / '.local/share/llm-serving-lab'
).resolve()
root = base / 'compiler-gcc14-r3-20261103'
packages = root / 'packages'
sysroot = root / 'sysroot'
assert root.parent == base
if root.exists() and not (root / 'owner.json').exists():
    raise RuntimeError('目录存在但无本包所有权')
root.mkdir(parents=True, exist_ok=True)
packages.mkdir(exist_ok=True)
sysroot.mkdir(exist_ok=True)
(root / 'owner.json').write_text(
    json.dumps(
        {
            'owner': 'llm-serving-lab R3',
            'purpose': 'project-local compiler, no system package changes',
        }
    )
)
groups = {
    '14.3.0-14ubuntu1': 'gcc-14-base cpp-14-x86-64-linux-gnu cpp-14 libgcc-14-dev gcc-14-x86-64-linux-gnu libstdc++-14-dev g++-14-x86-64-linux-gnu gcc-14 g++-14',
    '16-20260322-1ubuntu1': 'libcc1-0 libgomp1 libitm1 libasan8 liblsan0 libtsan2 libubsan1 libhwasan0 libquadmath0 libstdc++6 libgcc-s1',
    '0.27-1build1': 'libisl23',
    '1.3.1-3': 'libmpc3',
    '2.43-2ubuntu2.4': 'libc-dev-bin libc6-dev libc6',
    '7.0.0-42.42': 'linux-libc-dev',
    '1.4.3-1build1': 'rpcsvc-proto',
}
specs = [name + '=' + version for version, names in groups.items() for name in names.split()]
before = subprocess.check_output(['dpkg-query', '-W', '-f', '${Package}\t${Version}\n'])
expected = {}
for spec in specs:
    metadata = subprocess.check_output(['apt-cache', 'show', spec], text=True)
    item = {
        line.split(': ', 1)[0]: line.split(': ', 1)[1]
        for line in metadata.splitlines()
        if ': ' in line and not line.startswith(' ')
    }
    if item.get('Version') != spec.split('=', 1)[1] or not item.get('SHA256'):
        raise RuntimeError('apt metadata不匹配')
    expected[item['SHA256']] = {
        'package': spec,
        'bytes': int(item['Size']),
        'filename': item['Filename'],
    }
(root / 'package-manifest.json').write_text(json.dumps(expected, indent=2) + '\n')
subprocess.run(['apt-get', 'download', *specs], cwd=packages, check=True)
verified = []
for package in sorted(packages.glob('*.deb')):
    sha = hashlib.sha256(package.read_bytes()).hexdigest()
    if sha not in expected or package.stat().st_size != expected[sha]['bytes']:
        raise RuntimeError('下载包摘要不匹配')
    subprocess.run(['dpkg-deb', '-x', str(package), str(sysroot)], check=True)
    verified.append(sha)
assert set(verified) == set(expected) and len(verified) == 27
# Ubuntu的usrmerge由基础系统提供，不属于这些leaf包；只在私有sysroot补相对链接。
for name in ('lib', 'lib64', 'bin', 'sbin'):
    path = sysroot / name
    if not path.exists() and not path.is_symlink():
        path.symlink_to('usr/' + name, target_is_directory=True)
lib = sysroot / 'usr/lib/x86_64-linux-gnu'
cc1 = next(sysroot.rglob('cc1')).parent
gcc_lib = sysroot / 'usr/lib/gcc/x86_64-linux-gnu/14'
binary = root / 'bin'
binary.mkdir(exist_ok=True)
for name, compiler in [('cc', 'x86_64-linux-gnu-gcc-14'), ('c++', 'x86_64-linux-gnu-g++-14')]:
    executable = sysroot / 'usr/bin' / compiler
    assert executable.exists()
    script = (
        '#!/bin/sh\nexport LD_LIBRARY_PATH='
        + shlex.quote(str(lib))
        + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
    )
    args = [
        str(executable),
        '--sysroot=' + str(sysroot),
        '-B' + str(cc1) + '/',
        '-B' + str(gcc_lib) + '/',
        '-L' + str(lib),
        '-Wl,-rpath,' + str(lib),
    ]
    script += 'exec ' + ' '.join(shlex.quote(a) for a in args) + ' "$@"\n'
    wrapper = binary / name
    wrapper.write_text(script)
    wrapper.chmod(0o755)
smoke = root / 'smoke'
smoke.mkdir(exist_ok=True)
(smoke / 'c.c').write_text(
    '#include <Python.h>\n#include <stdio.h>\nint probe(void) { return 7; }\n'
)
(smoke / 'cpp.cpp').write_text(
    '#include <vector>\nextern "C" int probe(void) {std::vector<int> x={1,2,3}; return x[0]+x[1]+x[2];}\n'
)
for compiler, source, answer in [('cc', 'c.c', 7), ('c++', 'cpp.cpp', 6)]:
    output = smoke / (source + '.so')
    subprocess.run(
        [
            str(binary / compiler),
            '-shared',
            '-fPIC',
            '-I' + sysconfig.get_path('include'),
            str(smoke / source),
            '-o',
            str(output),
        ],
        check=True,
    )
    assert ctypes.CDLL(str(output)).probe() == answer
after = subprocess.check_output(['dpkg-query', '-W', '-f', '${Package}\t${Version}\n'])
assert before == after
receipt = {
    'root': str(root),
    'packages': len(verified),
    'package_bytes': sum(v['bytes'] for v in expected.values()),
    'system_packages_unchanged': True,
    'system_package_manifest_sha256': hashlib.sha256(before).hexdigest(),
    'cpu_compile_link_and_load': 'C and C++ passed',
    'cc': str(binary / 'cc'),
    'cxx': str(binary / 'c++'),
    'version': subprocess.check_output([str(binary / 'cc'), '--version'], text=True).splitlines()[
        0
    ],
    'binutils': subprocess.check_output(['ld', '--version'], text=True).splitlines()[0],
}
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
