"""固定上游路由的部署包装，仅接入本包证书，路由实现保持原字节。"""

import os
from pathlib import Path
import ssl

import aiohttp
import requests
from vllm_router import app
from vllm_router.aiohttp_client import AiohttpClientWrapper


def main():
    root = Path(os.environ['REFERENCE_SECRETS'])
    context = ssl.create_default_context(cafile=str(root / 'ca.pem'))
    context.load_cert_chain(str(root / 'client.pem'), str(root / 'client.key'))
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    class MutualTLSClient(AiohttpClientWrapper):
        def start(self):
            self.async_client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, ssl=context)
            )

    app.app.state.aiohttp_client_wrapper = MutualTLSClient()
    original_get = requests.get
    backends = tuple(os.environ['REFERENCE_BACKENDS'].split(','))

    def backend_get(url, *args, **kwargs):
        # 上游stats线程另用requests，证书配置必须覆盖这条只读旁路。
        if url not in tuple(endpoint + '/metrics' for endpoint in backends):
            raise ValueError('上游参考客户端访问了冻结后端之外的URL')
        kwargs['verify'] = str(root / 'ca.pem')
        kwargs['cert'] = (str(root / 'client.pem'), str(root / 'client.key'))
        return original_get(url, *args, **kwargs)

    requests.get = backend_get
    original_run = app.uvicorn.run

    def tls_run(application, **kwargs):
        kwargs.update(
            ssl_certfile=str(root / 'gateway-tls.pem'),
            ssl_keyfile=str(root / 'gateway-tls.key'),
            ssl_ca_certs=str(root / 'ca.pem'),
            ssl_cert_reqs=ssl.CERT_REQUIRED,
        )
        return original_run(application, **kwargs)

    app.uvicorn.run = tls_run
    app.main()


if __name__ == '__main__':
    main()
