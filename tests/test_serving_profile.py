"""配置快照和实际发送参数必须一致；不同用途不能相互继承资格。"""

import pytest


def test_profile_is_immutable_and_wire_budget_is_bounded():
    from serving_lab.profile import ServingProfile

    profile = ServingProfile({'name': 'test', 'max_tokens': 128, 'context_tokens': 4096}, 'model')
    original = profile.digest
    view = profile.document
    view['max_tokens'] = 999
    assert profile.digest == original and profile.document['max_tokens'] == 128
    body = {
        'model': 'model',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_tokens': 4,
        'stream': False,
    }
    payload = profile.payload(body)
    assert (
        payload['temperature'] == 0
        and payload['seed'] == 0
        and payload['chat_template_kwargs'] == {'enable_thinking': False}
    )
    with pytest.raises(ValueError):
        profile.payload({**body, 'max_tokens': 129})


def test_new_contract_rejects_old_runtime_before_recovery(tmp_path):
    from serving_lab.runtime import RuntimeStore
    from serving_lab.profile import check_runtime_contract

    with RuntimeStore(tmp_path) as store:
        store.record_arrival('untouched', None)
    before = (tmp_path / 'service.sqlite').read_bytes()
    with pytest.raises(ValueError):
        check_runtime_contract(tmp_path, 'version_transport_ready_v2')
    assert (tmp_path / 'service.sqlite').read_bytes() == before


def test_legacy_entry_cannot_recover_v2_runtime(tmp_path):
    from serving_lab.runtime import RuntimeStore
    from serving_lab.online import create_app
    from test_online import FakeEngine

    with RuntimeStore(tmp_path) as store:
        store.db.execute(
            'CREATE TABLE release_contract(id INTEGER PRIMARY KEY,scope TEXT,document TEXT)'
        )
        store.db.execute("INSERT INTO release_contract VALUES(1,'version_transport_ready_v2','{}')")
        store.db.commit()
        store.record_arrival('untouched', None)
    before = (tmp_path / 'service.sqlite').read_bytes()
    with pytest.raises(ValueError):
        create_app(FakeEngine(), tmp_path)
    assert (tmp_path / 'service.sqlite').read_bytes() == before
