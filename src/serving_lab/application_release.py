"""应用质量证据与请求边界；传输就绪、应用资格和生产验收分别判定。"""

import json
import re
from collections import Counter
from pathlib import Path
from types import MappingProxyType

from .execution import write_once
from .lab import canonical, digest, file_hash
from .profile import ServingProfile
from .profile import TRANSPORT_SCOPE
from .releases import ReleaseError
from .standard_actor import INSTRUCTION
from .task_protocol import FAMILIES

BUNDLE_NAME = 'c0_application_release_evidence_v1'
POOL_SCOPE = 'application_replica_pool_v1'
WORKER_SCOPE = 'profile_transport_and_choice_v1'


def policy_document():
    return {
        'qualification_scope': 'q2-coldstart-dev-controls-v1',
        'request_contract': 'public_choice_actor_v1',
        'cases': 24,
        'minimum_successes': 18,
        'minimum_family_successes': 2,
        'actor_sources': {
            n: file_hash(Path(__file__).with_name(n))
            for n in ('standard_actor.py', 'coldstart_actor.py', 'task_protocol.py')
        },
        'max_input_tokens': 3500,
        'max_tokens': 64,
        'context_tokens': 4096,
        'claim_scope': 'single-seed development qualification; trusted public-state caller; not arbitrary-chat or production guarantee',
    }


class ApplicationPolicy:
    __slots__ = ('_encoded', '_bindings', '_accepted', '_summaries', '_document')

    def __init__(self, bundle):
        if (
            bundle.get('schema') != 1
            or bundle.get('name') != BUNDLE_NAME
            or digest({k: v for k, v in bundle.items() if k != 'bundle_digest'})
            != bundle.get('bundle_digest')
            or bundle.get('policy') != policy_document()
        ):
            raise ValueError('应用证据包/actor合同不符')
        source = bundle.get('source_verification', {})
        source_pair = (source.get('producer'), source.get('evidence_level'))
        if source_pair not in {
            ('serving_lab.coldstart_release_import.v1', 'real_local_model_evaluation'),
            ('serving_lab.application_release_import.v2', 'real_local_and_cloud_model_evaluation'),
        }:
            raise ValueError('软件夹具或未核评价不能作为应用质量来源')
        if source_pair[0] == 'serving_lab.application_release_import.v2':
            # 保留旧证据包身份；新增云评价须逐工件绑定独立原件复核。
            added = source.get('added_audits')
            if (
                not re.fullmatch('[0-9a-f]{64}', str(source.get('parent_bundle_sha256')))
                or not isinstance(added, dict)
                or not added
                or any(
                    not re.fullmatch('[0-9a-f]{64}', str(value))
                    or bundle.get('certificates', {})
                    .get(artifact, {})
                    .get('source', {})
                    .get('raw_audit_sha256')
                    != value
                    for artifact, value in added.items()
                )
            ):
                raise ValueError('云评价没有绑定原应用包及逐工件审计')
        panel = bundle['interface_panel']
        if (
            panel.get('name') != 'public_command_interface_v1'
            or len(panel['cases']) != 4
            or len({c['id'] for c in panel['cases']}) != 4
        ):
            raise ValueError('命令接口探针必须完整')
        if not bundle['certificates']:
            raise ValueError('缺少应用评价')
        for artifact, cert in bundle['certificates'].items():
            if (
                artifact != cert['artifact_digest']
                or digest(cert['profile']) != cert['profile_digest']
                or cert['profile'].get('artifact_digest') != artifact
                or cert['profile'].get('name') != 'q2_public_choice_v1'
                or cert['profile'].get('max_tokens') != 64
                or cert['profile'].get('context_tokens') != 4096
                or cert['profile'].get('thinking') is not False
                or cert['profile'].get('engine_version') != '0.11.2'
            ):
                raise ValueError('应用评价未绑定原工件/profile')
            for key in ('artifact_digest', 'parameter_digest', 'profile_digest'):
                if not re.fullmatch('[0-9a-f]{64}', cert[key]):
                    raise ValueError('应用身份摘要无效')
            cases = cert['case_results']
            expected_ids = {f'cs_dev_{f + 1:02}_{n + 1:02}' for f in range(6) for n in range(4)}
            if (
                len(cases) != 24
                or {r['task_id'] for r in cases} != expected_ids
                or Counter(r['family'] for r in cases) != Counter({f: 4 for f in FAMILIES})
                or any(
                    type(r['success']) is not bool
                    or r['family'] != FAMILIES[int(r['task_id'].split('_')[2]) - 1]
                    for r in cases
                )
            ):
                raise ValueError('应用评价分母/任务族不完整')
            # 此接口使用冻结profile；请求不能覆盖采样，也不能以空探针回避命令约束。
            p = ServingProfile(cert['profile'], 'probe')
            for case in panel['cases']:
                if not case['messages'] or not all(
                    set(m) == {'role', 'content'} and isinstance(m['content'], str)
                    for m in case['messages']
                ):
                    raise ValueError('接口探针消息无效')
                p.payload(
                    {
                        'messages': case['messages'],
                        'max_tokens': 64,
                        'stream': False,
                        'action_choices': case['choices'],
                    }
                )
        object.__setattr__(self, '_encoded', canonical(bundle))
        document = {
            **bundle['policy'],
            'bundle_digest': bundle['bundle_digest'],
            'interface_panel_digest': digest(panel),
        }
        summaries = {}
        accepted = set()
        bindings = {}
        for artifact, cert in bundle['certificates'].items():
            successes = sum(r['success'] for r in cert['case_results'])
            families = Counter(r['family'] for r in cert['case_results'] if r['success'])
            passed = successes >= 18 and all(families[f] >= 2 for f in FAMILIES)
            if passed:
                accepted.add(artifact)
            bindings[artifact] = (cert['parameter_digest'], cert['profile_digest'])
            summaries[artifact] = canonical(
                {
                    'qualification_scope': document['qualification_scope'],
                    'passed': passed,
                    'successes': successes,
                    'cases': 24,
                    'family_successes': {f: families[f] for f in FAMILIES},
                    'evidence_digest': digest(cert),
                    'bundle_digest': bundle['bundle_digest'],
                    'actor_digest': digest(document['actor_sources']),
                }
            )
        object.__setattr__(self, '_bindings', MappingProxyType(bindings))
        object.__setattr__(self, '_accepted', frozenset(accepted))
        object.__setattr__(self, '_summaries', MappingProxyType(summaries))
        object.__setattr__(self, '_document', canonical(document))

    def __setattr__(self, *args):
        raise AttributeError('应用策略不可变')

    @property
    def bundle(self):
        return json.loads(self._encoded)

    @property
    def document(self):
        return json.loads(self._document)

    def summary(self, cert):
        return json.loads(self._summaries[cert['artifact_digest']])

    def bind(self, engines, parameters):
        result = {}
        certificates = self.bundle['certificates']
        for engine in engines.values():
            cert = certificates.get(engine.binding.artifact_digest)
            if (
                not cert
                or parameters[engine.version_id]['digest'] != cert['parameter_digest']
                or engine.profile.digest != cert['profile_digest']
                or engine.profile.document != cert['profile']
            ):
                raise ValueError('应用证据与实际参数、工件或profile不匹配')
            result[engine.version_id] = self.summary(cert)
        return result

    def require(self, version_document):
        artifact = version_document['artifact_digest']
        # 判定在启动时完成，请求路径只比较不可变身份，不重算评价面板。
        if artifact not in self._accepted or self._bindings.get(artifact) != (
            version_document['parameter_digest'],
            version_document['profile_digest'],
        ):
            raise ReleaseError('application_quality_rejected', 409)

    def validate_request(self, body):
        messages = body['messages']
        choices = body.get('action_choices')
        if (
            body['max_tokens'] != 64
            or not isinstance(choices, list)
            or not 1 <= len(choices) <= 256
            or any(
                not isinstance(c, str)
                or not c
                or len(c) > 256
                or c != c.strip()
                or '\n' in c
                or '\r' in c
                for c in choices
            )
            or choices != sorted(set(choices))
            or not {'look', 'inventory'} <= set(choices)
            or len(messages) != 2
            or messages[0] != {'role': 'system', 'content': INSTRUCTION}
            or messages[1]['role'] != 'user'
        ):
            raise ValueError('请求不符合已评价actor的消息/命令合同')
        doc = json.loads(messages[1]['content'])
        keys = {
            'GOAL',
            'INITIAL_OBSERVATION',
            'PAST_PUBLIC_EVIDENCE_MAY_BE_STALE',
            'RECENT_ACTION_FEEDBACK',
            'CURRENT_FEEDBACK',
            'CURRENT_COMMANDS',
        }
        if (
            not isinstance(doc, dict)
            or set(doc) != keys
            or canonical(doc) != messages[1]['content']
            or doc['CURRENT_COMMANDS'] != choices
        ):
            raise ValueError('公开状态结构或完整候选集不匹配')
        if not all(
            isinstance(doc[k], str) for k in ('GOAL', 'INITIAL_OBSERVATION', 'CURRENT_FEEDBACK')
        ):
            raise ValueError('公开文本类型不符')
        goal = re.search(r'Your task is to:\s*([^\n]+)', doc['INITIAL_OBSERVATION'])
        if not goal or doc['GOAL'] != goal[1].strip():
            raise ValueError('目标与原始公开观察不一致')
        state = doc['PAST_PUBLIC_EVIDENCE_MAY_BE_STALE']
        fields = {
            'visited_places': {'place', 'step'},
            'previously_seen_take_options': {'command', 'step'},
            'interaction_evidence': {'action', 'feedback', 'step'},
        }
        if (
            not isinstance(state, dict)
            or set(state) != set(fields)
            or not isinstance(doc['RECENT_ACTION_FEEDBACK'], list)
            or len(doc['RECENT_ACTION_FEEDBACK']) > 8
        ):
            raise ValueError('公开工作状态结构不符')
        for name, required in fields.items():
            if not isinstance(state[name], list):
                raise ValueError('状态必须为条目列表')
            for row in state[name]:
                self._evidence_row(row, required)
        for row in doc['RECENT_ACTION_FEEDBACK']:
            self._evidence_row(row, {'step', 'action', 'feedback'})

    @staticmethod
    def _evidence_row(row, required):
        if (
            not isinstance(row, dict)
            or set(row) != required
            or type(row['step']) is not int
            or row['step'] < 0
            or any(not isinstance(row[k], str) for k in required - {'step'})
        ):
            raise ValueError('公开证据条目不符')


def load_policy(path, expected_file_sha256):
    if file_hash(path) != expected_file_sha256:
        raise ValueError('应用证据文件变化，不能沿用发布配置')
    return ApplicationPolicy(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_transport_panel(panel):
    """先核固定调用上限和四种探针用途，再允许触碰模型或runtime。"""
    cases = panel.get('cases')
    if (
        panel.get('qualification_scope') != TRANSPORT_SCOPE
        or panel.get('required_passes') != 4
        or not isinstance(cases, list)
        or len(cases) != 4
        or len({c.get('id') for c in cases}) != 4
    ):
        raise ValueError('应用worker要求完整固定4项传输面板')
    if Counter((c.get('stream'), c.get('expected_finish')) for c in cases) != Counter(
        {(False, 'stop'): 2, (True, 'stop'): 1, (False, 'length'): 1}
    ):
        raise ValueError('传输面板必须覆盖普通、重复、流式和预算截断')
    for case in cases:
        if (
            type(case.get('stream')) is not bool
            or type(case.get('max_tokens')) is not int
            or not 1 <= case['max_tokens'] <= 64
            or not isinstance(case.get('messages'), list)
            or not case['messages']
            or any(
                set(m) != {'role', 'content'}
                or m['role'] not in {'system', 'user', 'assistant'}
                or not isinstance(m['content'], str)
                for m in case['messages']
            )
        ):
            raise ValueError('传输探针消息或预算不符')
    if (
        type(panel.get('deadline_seconds')) not in {int, float}
        or not 0 < panel['deadline_seconds'] <= 90
    ):
        raise ValueError('传输探针期限不符')


async def qualify_application_worker(
    engine, panel, output, store, policy, application_result, cleanups=None
):
    from .transport import qualify_transport
    from .research_action import generate_action

    validate_transport_panel(panel)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    transport = await qualify_transport(
        engine,
        panel,
        out / 'transport',
        store=store,
        cleanups=cleanups,
        application_result=application_result,
    )
    rows = []
    if transport['passed']:
        for case in policy.bundle['interface_panel']['cases']:
            ids = await engine.token_ids(case['messages'])
            receipt = await generate_action(
                engine,
                store,
                out / 'interface' / case['id'],
                case['messages'],
                ids,
                {
                    'kind': 'application_interface_qualification',
                    'case': case['id'],
                    'qualification_scope': WORKER_SCOPE,
                },
                choices=case['choices'],
            )
            rows.append(
                {
                    'case': case['id'],
                    'passed': receipt['finish_reason'] == 'stop'
                    and receipt['content'] in case['choices'],
                    'receipt': receipt,
                }
            )
            if not rows[-1]['passed']:
                break
    interface_passed = len(rows) == 4 and all(r['passed'] for r in rows)
    result = {
        'qualification_scope': WORKER_SCOPE,
        'passed': transport['passed'] and interface_passed,
        'version_id': engine.version_id,
        'worker': engine.identity,
        'artifact_digest': engine.binding.artifact_digest,
        'profile_digest': engine.profile.digest,
        'panel_digest': digest(panel),
        'application_history': application_result,
        'transport_passed': transport['passed'],
        'interface_passed': interface_passed,
        'transport_cases': transport['cases'],
        'interface_cases': len(rows),
        'transport_receipt': transport,
        'interface_results': rows,
        'meaning': 'worker transport and command interface only; model application evidence is independently gated',
    }
    write_once(out / 'receipt.json', result)
    return result
