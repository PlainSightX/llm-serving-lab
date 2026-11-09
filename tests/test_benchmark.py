"""测量口径反例：role帧不是首token，失败不能从计划分母消失。"""


async def test_first_content_not_role_and_missing_done_is_failure():
    from serving_lab.benchmark import StreamObservation

    observed = StreamObservation('m', 2, 8)
    observed.feed(
        '{"model":"m","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}', 0.1
    )
    assert observed.first_content is None
    observed.feed('{"model":"m","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}', 0.3)
    observed.feed(
        '{"model":"m","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
        0.4,
    )
    assert observed.first_content == 0.3 and not observed.valid
    observed.feed('[DONE]', 0.5)
    assert observed.valid


def test_rejections_and_unsent_requests_stay_in_denominator():
    from serving_lab.benchmark import summarize

    rows = [
        {
            'protocol_valid': True,
            'sent': True,
            'status': 200,
            'ttft_scheduled': 0.2,
            'e2e_scheduled': 1,
            'output_budget': 16,
            'send_lag': 0.001,
            'ttft_sent': 0.199,
            'usage': {'completion_tokens': 5},
            'expected_version': 'm',
            'shape': 'short-short',
        },
        {'protocol_valid': False, 'sent': True, 'status': 429, 'send_lag': 0.002},
        {'protocol_valid': False, 'sent': False, 'status': None, 'send_lag': 0.3},
    ]
    result = summarize(
        rows, 3, {'ttft_seconds': 1, 'short_e2e_seconds': 5, 'long_e2e_seconds': 10}, interval=1
    )
    assert result['planned'] == 3 and result['sent'] == 2 and result['protocol_valid'] == 1
    assert (
        result['slo_met'] == 1
        and result['slo_fraction'] == 1 / 3
        and result['slo_goodput_rps'] == 1 / 3
    )


def test_metric_labels_and_invalid_baselines_are_not_silently_combined():
    from serving_lab.benchmark import metric_delta, metric_series_delta

    before = {'vllm:request_success_total{finished_reason="stop"}': 4}
    after = {**before, 'vllm:request_success_total{finished_reason="abort"}': 1}
    result = metric_series_delta(before, after, 'vllm:request_success_total')
    assert result['vllm:request_success_total{finished_reason="abort"}']['delta'] is None
    assert metric_delta(before, after, 'vllm:request_success_total') is None
    assert metric_delta(after, before, 'vllm:request_success_total') is None


async def test_open_loop_sends_next_request_before_previous_finishes(monkeypatch):
    import asyncio
    import httpx
    from serving_lab.benchmark import window

    arrivals = []
    original = httpx.AsyncClient

    async def handler(request):
        arrivals.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.1)
        return httpx.Response(429, text='busy')

    monkeypatch.setattr(
        httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    requests = [
        {
            'id': str(i),
            'shape': 's',
            'expected_version': 'm',
            'input_tokens': 2,
            'output_budget': 16,
            'body': {},
            'headers': {},
        }
        for i in range(2)
    ]
    result = await window(
        'http://test', requests, [0, 0.02], {'client_timeout_seconds': 1, 'late_drop_seconds': 0.25}
    )
    assert len(arrivals) == 2 and arrivals[1] - arrivals[0] < 0.08
    assert len(result['rows']) == 2 and all(row['status'] == 429 for row in result['rows'])


async def test_window_supports_fixed_per_request_endpoints_without_changing_default(monkeypatch):
    import httpx
    from serving_lab.benchmark import window

    targets = []
    original = httpx.AsyncClient

    def handler(request):
        targets.append(str(request.url))
        return httpx.Response(429, text='busy')

    monkeypatch.setattr(
        httpx,
        'AsyncClient',
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    base = {
        'shape': 's',
        'expected_version': 'm',
        'input_tokens': 2,
        'output_budget': 64,
        'body': {},
        'headers': {},
    }
    requests = [
        {**base, 'id': 'default'},
        {**base, 'id': 'other', 'endpoint': 'https://worker-b:30446/'},
    ]
    await window(
        'https://gateway:30443',
        requests,
        [0, 0.01],
        {'client_timeout_seconds': 1, 'late_drop_seconds': 0.25},
    )
    assert targets == [
        'https://gateway:30443/v1/chat/completions',
        'https://worker-b:30446/v1/chat/completions',
    ]


def test_one_unsent_request_invalidates_generator_even_with_small_p95_lag():
    from serving_lab.benchmark import summarize

    row = {
        'protocol_valid': True,
        'sent': True,
        'status': 200,
        'ttft_scheduled': 0.1,
        'ttft_sent': 0.099,
        'e2e_scheduled': 0.2,
        'output_budget': 16,
        'send_lag': 0.001,
        'usage': {'completion_tokens': 1},
    }
    rows = [dict(row) for _ in range(31)] + [
        {'protocol_valid': False, 'sent': False, 'status': None, 'send_lag': 0.3}
    ]
    result = summarize(
        rows, 32, {'ttft_seconds': 1, 'short_e2e_seconds': 5, 'long_e2e_seconds': 10}, interval=1
    )
    assert result['send_lag_p95'] == 0.001 and not result['schedule_valid']


async def test_publication_error_retains_completed_request_rows(monkeypatch):
    import httpx
    from serving_lab.benchmark import window

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        'AsyncClient',
        lambda **kw: original(
            transport=httpx.MockTransport(lambda r: httpx.Response(429, text='busy')), **kw
        ),
    )
    requests = [
        {
            'id': str(i),
            'shape': 's',
            'expected_version': 'm',
            'input_tokens': 2,
            'output_budget': 16,
            'body': {},
            'headers': {},
        }
        for i in range(2)
    ]

    async def failed(started):
        raise RuntimeError('publication failed')

    result = await window(
        'http://test',
        requests,
        [0, 0.01],
        {'client_timeout_seconds': 1, 'late_drop_seconds': 0.25},
        action=failed,
    )
    assert len(result['rows']) == 2 and result['action']['error']['type'] == 'RuntimeError'


def test_allowed_version_during_late_switch_is_not_protocol_corruption():
    from serving_lab.benchmark import StreamObservation

    observed = StreamObservation('b', 2, 8, ['a', 'b'])
    observed.feed(
        '{"model":"a","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
        0.1,
    )
    observed.feed('[DONE]', 0.2)
    assert observed.valid and observed.actual_model == 'a'
