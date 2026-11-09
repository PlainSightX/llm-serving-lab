"""部署边界的TLS配置。调用方显式提供信任根和自身证书。"""

from pathlib import Path
import ssl


def _files(material):
    if set(material) != {'ca', 'cert', 'key'}:
        raise ValueError('TLS配置需要ca、cert、key三个文件')
    paths = {}
    for name, value in material.items():
        if not isinstance(value, str) or not value or not Path(value).is_file():
            raise ValueError(f'TLS文件不可用: {name}')
        paths[name] = str(Path(value).resolve())
    return paths


def client_context(material):
    """保留CA及主机名校验，同时发送调用方证书。"""
    paths = _files(material)
    context = ssl.create_default_context(cafile=paths['ca'])
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(paths['cert'], paths['key'])
    return context


def listener_options(host, port, *, ca=None, cert=None, key=None):
    """原loopback入口保持兼容，网络监听必须校验客户端证书。"""
    options = {'host': host, 'port': port}
    supplied = any(value is not None for value in (ca, cert, key))
    if not supplied and host in {'127.0.0.1', 'localhost', '::1'}:
        return options
    paths = _files({'ca': ca, 'cert': cert, 'key': key})
    options.update(
        ssl_certfile=paths['cert'],
        ssl_keyfile=paths['key'],
        ssl_ca_certs=paths['ca'],
        ssl_cert_reqs=ssl.CERT_REQUIRED,
    )
    return options
