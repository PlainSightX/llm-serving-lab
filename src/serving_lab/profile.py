"""不可变的有效服务配置；与参数身份分开，策略变化不能沿用旧资格。"""

from __future__ import annotations
import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from .lab import canonical, digest, file_hash

TRANSPORT_SCOPE = 'version_transport_ready_v2'
LEGACY_SCOPE = 'application_exact_v1'


class ServingProfile:
    __slots__ = ('_encoded', '_digest', '_model')

    def __init__(self, document, model):
        object.__setattr__(self, '_encoded', canonical(document))
        object.__setattr__(self, '_digest', digest(document))
        object.__setattr__(self, '_model', model)

    def __setattr__(self, *args):
        raise AttributeError('服务配置不可变')

    @property
    def document(self):
        return json.loads(self._encoded)

    @property
    def digest(self):
        return self._digest

    def payload(self, body):
        choice_mode = self.document.get('name') == 'q2_public_choice_v1'
        length_mode = self.document.get('name') == 'n2_length_controlled_v1'
        allowed = (
            {'model', 'messages', 'max_tokens', 'stream'}
            | ({'action_choices'} if choice_mode else set())
            | ({'ignore_eos'} if length_mode else set())
        )
        if set(body) - allowed:
            raise ValueError('未经批准的服务配置覆盖')
        if 'ignore_eos' in body and type(body['ignore_eos']) is not bool:
            raise ValueError('固定长度开关必须为bool')
        request = copy.deepcopy(body)
        choices = request.pop('action_choices', None)
        if 'action_choices' in body:
            if (
                not isinstance(choices, list)
                or not 1 <= len(choices) <= 256
                or not all(
                    isinstance(c, str)
                    and c == c.strip()
                    and 0 < len(c) <= 256
                    and '\n' not in c
                    and '\r' not in c
                    for c in choices
                )
                or len(set(choices)) != len(choices)
            ):
                raise ValueError('无效动作候选集合')
        maximum = body.get('max_tokens', self.document['max_tokens'])
        if type(maximum) is not int or not 1 <= maximum <= self.document['max_tokens']:
            raise ValueError('输出预算超过profile')
        payload = {
            **request,
            'model': self._model,
            'max_tokens': maximum,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': -1,
            'min_p': 0.0,
            'seed': 0,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'repetition_penalty': 1.0,
            'stop': [],
            'stop_token_ids': [],
            'ignore_eos': False,
            'skip_special_tokens': True,
            'include_stop_str_in_output': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        if payload.get('stream'):
            payload['stream_options'] = {'include_usage': True}
        payload.update(self.document.get('decoding', {}))
        if length_mode:
            payload['ignore_eos'] = body.get('ignore_eos', False)
        if choices is not None:
            payload['structured_outputs'] = {'choice': choices}
        return payload


def make_profile(binding, tokenizer, policy=None):
    launch = json.loads((binding.manifest_path.parent / 'launch.json').read_text())
    config = launch['config']
    model = json.loads((Path(binding.model_path) / 'config.json').read_text())
    if (
        '--generation-config' not in launch['command']
        or launch['command'][launch['command'].index('--generation-config') + 1] != 'vllm'
    ):
        raise ValueError('profile要求显式vllm generation config')
    template = tokenizer.get_chat_template()
    document = {
        'schema': 1,
        'name': 'greedy_non_thinking_v1',
        'engine_version': launch['versions']['vllm'],
        'artifact_digest': binding.artifact_digest,
        'dtype': config['dtype'],
        'quantization': model.get('quantization_config'),
        'template_sha256': hashlib.sha256(template.encode()).hexdigest(),
        'eos_token_id': tokenizer.eos_token_id,
        'model_eos_token_id': model.get('eos_token_id'),
        'thinking': False,
        'context_tokens': config['max_model_len'],
        'max_tokens': 128,
        'request_overrides': ['messages', 'max_tokens within1..128', 'stream'],
        'output_constraints': None,
        'implementation': {
            name: file_hash(Path(__file__).with_name(name)) for name in ('profile.py', 'engine.py')
        },
    }
    # 同一函数产生配置说明与wire；完整有效解码值进入身份，避免说明与实际覆盖分离。
    if policy == {'name': 'n2_length_controlled_v1', 'seed': 0}:
        document.update(
            name=policy['name'],
            max_tokens=512,
            request_overrides=[
                'messages',
                'max_tokens within1..512',
                'stream',
                'ignore_eos bool for length-controlled measurement',
            ],
        )
    elif policy is not None:
        legacy = policy == {'name': 'q2_public_non_thinking_v1', 'seed': 20261103}
        rescue = (
            set(policy) == {'name', 'seed'}
            and policy['name'] == 'q2_public_non_thinking_v2'
            and type(policy['seed']) is int
            and policy['seed'] in {20261103, 20261104, 20261105}
        )
        choice = policy == {'name': 'q2_public_choice_v1', 'seed': 20261103}
        if not (legacy or rescue or choice):
            raise ValueError('未知研究解码策略')
        document.update(
            name=policy['name'],
            max_tokens=64,
            request_overrides=['messages', 'max_tokens within1..64', 'stream'],
        )
        document['decoding'] = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'seed': policy['seed'],
        }
        if choice:
            document['request_overrides'].append(
                'action_choices: 1..256 unique single-line commands, <=256 characters each'
            )
            document['output_constraints'] = {
                'kind': 'public_command_choice_v1',
                'wire_field': 'structured_outputs.choice',
                'transport_probe': 'unconstrained; structured canaries required before task calls',
                'actor_requirement': 'all current public commands plus look/inventory; no answer-based filtering',
            }
    sample = ServingProfile(document, binding.served_model).payload(
        {
            'model': binding.served_model,
            'messages': [],
            'max_tokens': document['max_tokens'],
            'stream': False,
        }
    )
    document['decoding'] = {
        k: v for k, v in sample.items() if k not in {'model', 'messages', 'max_tokens', 'stream'}
    }
    return ServingProfile(document, binding.served_model)


def check_runtime_contract(root, scope, expected=None):
    """必须在RuntimeStore.recover之前调用，错误入口不能触碰原在途记录。"""
    path = Path(root).resolve() / 'service.sqlite'
    if not path.exists():
        return
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='release_contract'").fetchone()
        if not exists:
            if scope != LEGACY_SCOPE:
                raise ValueError('旧runtime缺少新合同身份，拒绝升级')
            return
        row = db.execute('SELECT scope,document FROM release_contract WHERE id=1').fetchone()
        if not row or row[0] != scope:
            raise ValueError('runtime合同不同，须使用新目录')
        if expected is not None and json.loads(row[1]) != expected:
            raise ValueError('runtime配置身份不同，须使用新目录')


def application_history(path, binding):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (
        value['version_id'] != binding.version_id
        or value['artifact_digest'] != binding.artifact_digest
        or value['panel_digest']
        != '9b652ffa4a993f6855ab1a7b9b05c3b839c3339a0a740ee565ff5f99e86cb0f4'
        or value['passed'] is not False
        or value['passed_cases'] != 3
        or value['cases'] != 8
    ):
        raise ValueError('未匹配当前工件的原v1失败记录')
    return {
        'scope': 'r2-service-functional-quality-v1',
        'status': 'failed',
        'passed_cases': 3,
        'cases': 8,
        'receipt_sha256': file_hash(path),
        'panel_digest': value['panel_digest'],
        'meaning': 'historical v1 conditions; not a test of this new profile',
    }
