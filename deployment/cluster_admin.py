"""从文件读取双向TLS材料和管理密钥，操作既有发布管理接口。"""

import argparse
import json
from pathlib import Path
import ssl
import urllib.request
from urllib.parse import urlsplit


def request(endpoint, secrets, path, body=None):
    url = urlsplit(endpoint)
    if (
        url.scheme != 'https'
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ('', '/')
    ):
        raise ValueError('管理入口需要纯HTTPS地址')
    root = Path(secrets)
    context = ssl.create_default_context(cafile=str(root / 'ca.pem'))
    context.load_cert_chain(str(root / 'client.pem'), str(root / 'client.key'))
    token = (root / 'admin-token').read_text(encoding='utf-8').strip()
    data = json.dumps(body).encode('utf-8') if body is not None else None
    call = urllib.request.Request(
        endpoint.rstrip('/') + path,
        data=data,
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
    )
    with opener.open(call, timeout=120) as response:
        return json.load(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--secrets', required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('status')
    for name in ('qualify', 'drain', 'reload', 'reconcile'):
        sub.add_parser(name).add_argument('replica', choices=['a', 'b'])
    route = sub.add_parser('route')
    route.add_argument('operation', choices=['promote', 'withdraw', 'rollback', 'canary'])
    route.add_argument('--version', required=True)
    route.add_argument('--generation', required=True, type=int)
    route.add_argument('--percentage', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'status':
        result = request(args.endpoint, args.secrets, '/admin/releases')
    elif args.action == 'route':
        result = request(
            args.endpoint,
            args.secrets,
            '/admin/route',
            {
                'operation': args.operation,
                'target': args.version,
                'expected_generation': args.generation,
                'percentage': args.percentage,
            },
        )
    else:
        result = request(
            args.endpoint, args.secrets, f'/admin/replicas/{args.replica}/{args.action}', {}
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
