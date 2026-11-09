"""断点续传仍须核验完整摘要，错误范围不能写进已有前缀。"""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
from pathlib import Path
from threading import Thread

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@pytest.mark.parametrize('known_size', [True, False])
def test_resume_uses_original_prefix_and_full_hash(tmp_path, known_size):
    from fetch_runtime_wheels import download

    payload = b'locked-runtime-content'
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = self.headers['Range']
            ranges.append(value)
            start = int(value.split('=')[1].split('-')[0])
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{len(payload) - 1}/{len(payload)}')
            self.send_header('Content-Length', str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    item = {
        'name': 'test-only',
        'file': 'test.whl',
        'url': f'http://127.0.0.1:{server.server_port}/wheel',
        'sha256': hashlib.sha256(payload).hexdigest(),
        'bytes': len(payload) if known_size else None,
    }
    (tmp_path / 'test.whl.partial').write_bytes(payload[:4])
    try:
        result = download(tmp_path, item)
        assert ranges[0].startswith('bytes=4-')
        assert (
            result['actual_bytes'] == len(payload)
            and (tmp_path / 'test.whl').read_bytes() == payload
        )
        bad = {**item, 'file': 'bad.whl', 'sha256': '0' * 64}
        with pytest.raises(ValueError, match='摘要'):
            download(tmp_path, bad)
        assert not (tmp_path / 'bad.whl').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
