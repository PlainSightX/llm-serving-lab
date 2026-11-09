"""临时测量入口只做双向TLS字节转发，后端限定同Pod的原生loopback。"""

import argparse
import asyncio
from contextlib import suppress
from pathlib import Path
import ssl


async def copy_stream(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()


async def relay(reader, writer):
    upstream = None
    tasks = []
    try:
        native_reader, upstream = await asyncio.wait_for(
            asyncio.open_connection('127.0.0.1', 8000), 5
        )
        tasks = [
            asyncio.create_task(copy_stream(reader, upstream)),
            asyncio.create_task(copy_stream(native_reader, writer)),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in (upstream, writer):
            if stream is not None:
                stream.close()
                with suppress(ConnectionError, ssl.SSLError):
                    await stream.wait_closed()


async def run(secrets, port):
    root = Path(secrets)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=str(root / 'ca.pem'))
    context.load_cert_chain(str(root / 'tls.pem'), str(root / 'tls.key'))
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    server = await asyncio.start_server(relay, '0.0.0.0', port, ssl=context)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--secrets', default='/secrets')
    parser.add_argument('--port', type=int, default=8444)
    args = parser.parse_args()
    asyncio.run(run(args.secrets, args.port))
