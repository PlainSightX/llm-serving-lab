"""固定上游router的独立CPU HTTP对照；backend是明确协议夹具。"""

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from serving_lab.engine import check_port_available, process_identity
from serving_lab.paths import lab_root
from serving_lab.execution import write_once


def run(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    servers = []
    threads = []
    calls = []
    lock = threading.Lock()
    router = None
    result = None

    def handler(label):
        class Backend(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def send(self, value, status=200):
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/v1/models':
                    self.send(
                        {
                            'object': 'list',
                            'data': [
                                {
                                    'id': 'contract-v1',
                                    'object': 'model',
                                    'owned_by': 'contract-fixture',
                                }
                            ],
                        }
                    )
                elif self.path == '/metrics':
                    raw = b'# contract fixture\n'
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self.send({'error': 'fixture_not_found'}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    calls.append({'backend': label, 'path': self.path, 'body': body})
                mode = body['messages'][0]['content']
                if mode == 'fail-before-headers':
                    self.close_connection = True
                    return
                if body.get('stream'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    self.wfile.write(
                        (
                            'data: '
                            + json.dumps(
                                {
                                    'model': 'contract-v1',
                                    'choices': [
                                        {'delta': {'content': label}, 'finish_reason': None}
                                    ],
                                }
                            )
                            + '\n\n'
                        ).encode()
                    )
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.send(
                    {
                        'id': 'fixture-' + label,
                        'object': 'chat.completion',
                        'model': 'contract-v1',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': label},
                                'finish_reason': 'stop',
                            }
                        ],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                    }
                )

        return Backend

    try:
        for port in (19190, 19191, 19192):
            check_port_available('127.0.0.1', port)
        for label, port in [('a', 19191), ('b', 19192)]:
            server = ThreadingHTTPServer(('127.0.0.1', port), handler(label))
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        executable = str(lab_root() / 'upstream-router-venv/bin/vllm-router')
        command = [
            executable,
            '--host',
            '127.0.0.1',
            '--port',
            '19190',
            '--service-discovery',
            'static',
            '--static-backends',
            'http://127.0.0.1:19191,http://127.0.0.1:19192',
            '--static-models',
            'contract-v1,contract-v1',
            '--routing-logic',
            'roundrobin',
            '--max-instance-failover-reroute-attempts',
            '0',
            '--engine-stats-interval',
            '30',
            '--feature-gates',
            'SemanticCache=false',
        ]
        env = dict(os.environ)
        env.update(
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(out / 'empty-hf-cache')
        )
        write_once(
            out / 'input.json',
            {
                'upstream_commit': 'e8cb4959ebfa333714ef468a3aefe158a65a09e6',
                'command': command,
                'scope': 'CPU contract fixtures, not real model quality or performance',
            },
        )
        with (out / 'router.log').open('xb') as log:
            router = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write_once(out / 'router-process.json', process_identity(router.pid))
        with httpx.Client(
            base_url='http://127.0.0.1:19190',
            timeout=10,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            for _ in range(200):
                if router.poll() is not None:
                    raise RuntimeError('upstream router startup failed')
                try:
                    r = client.get('/v1/models')
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError('upstream startup timeout')
            base = {
                'model': 'contract-v1',
                'messages': [{'role': 'user', 'content': 'hello'}],
                'max_tokens': 4,
                'stream': False,
            }
            responses = []
            for _ in range(6):
                r = client.post('/v1/chat/completions', json=base)
                r.raise_for_status()
                responses.append(r.json())
            chosen = [r['choices'][0]['message']['content'] for r in responses]
            assert chosen.count('a') == chosen.count('b') == 3
            before = len(calls)
            unknown = client.post('/v1/chat/completions', json={**base, 'model': 'not-registered'})
            assert 400 <= unknown.status_code < 500 and len(calls) == before
            try:
                failed = client.post(
                    '/v1/chat/completions',
                    json={**base, 'messages': [{'role': 'user', 'content': 'fail-before-headers'}]},
                )
                assert failed.status_code >= 500
                failure = {'status': failed.status_code, 'body': failed.text}
            except httpx.TransportError as exc:
                failure = {'status': None, 'transport_error': type(exc).__name__}
            assert len(calls) == before + 1
            before = len(calls)
            stream = client.post('/v1/chat/completions', json={**base, 'stream': True})
            assert (
                stream.status_code == 200
                and '[DONE]' not in stream.text
                and len(calls) == before + 1
            )
            write_once(
                out / 'responses.json',
                {
                    'normal': responses,
                    'unknown': {'status': unknown.status_code, 'body': unknown.text},
                    'connection_failure': failure,
                    'incomplete_stream': {'status': stream.status_code, 'body': stream.text},
                },
            )
            result = {
                'passed': True,
                'calls': len(calls),
                'roundrobin': chosen,
                'unknown_model_rejected': True,
                'connection_error_not_retried': True,
                'partial_stream_not_retried': True,
                'missing_DONE_preserved': True,
                'scope': 'independent upstream CPU routing contract only',
                'weight_or_tokenizer_downloads': list((out / 'empty-hf-cache').rglob('*'))
                if (out / 'empty-hf-cache').exists()
                else [],
            }
            assert not result['weight_or_tokenizer_downloads']
    finally:
        if router and router.poll() is None:
            router.terminate()
            try:
                router.wait(timeout=10)
            except subprocess.TimeoutExpired:
                router.kill()
                router.wait(timeout=5)
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(3)
        write_once(out / 'backend-calls.json', calls)
        write_once(
            out / 'cleanup.json',
            {
                'router_returncode': router.returncode if router else None,
                'fixture_threads_stopped': all(not t.is_alive() for t in threads),
            },
        )
        assert all(not t.is_alive() for t in threads)
    if result:
        write_once(out / 'summary.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))
