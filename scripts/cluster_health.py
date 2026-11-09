"""容器exec探针使用本容器证书，检查实际TLS接口的就绪状态。"""

import argparse
from pathlib import Path
import sys

import httpx
from serving_lab.tls import client_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['worker', 'gateway'])
    parser.add_argument('--secrets', default='/secrets')
    parser.add_argument('--port', type=int, default=8443)
    args = parser.parse_args()
    root = Path(args.secrets)
    prefix = 'gateway-' if args.role == 'gateway' else ''
    material = {
        'ca': str(root / 'ca.pem'),
        'cert': str(root / (prefix + 'tls.pem')),
        'key': str(root / (prefix + 'tls.key')),
    }
    token_name = 'admin-token' if args.role == 'gateway' else 'worker-token'
    path = '/admin/releases' if args.role == 'gateway' else '/identity'
    try:
        token = (root / token_name).read_text().strip()
        with httpx.Client(verify=client_context(material), trust_env=False, timeout=2) as client:
            response = client.get(
                f'https://localhost:{args.port}{path}', headers={'Authorization': 'Bearer ' + token}
            )
            response.raise_for_status()
            value = response.json()
        if args.role == 'worker' and not value['ready']:
            return 1
        return 0
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        return 1


if __name__ == '__main__':
    sys.exit(main())
