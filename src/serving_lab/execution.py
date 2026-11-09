"""真实执行回执合同。只有受控运行器调用写入函数，没有手工回执导入命令。"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from .lab import LabError, canonical, check_output, digest, file_hash, identifier

SOURCE_BYTES = Path(__file__).read_bytes()


DEFAULT_GENERATION = {
    'max_new_tokens': 128,
    'max_input_tokens': 1024,
    'seed': 42,
    'do_sample': False,
    'device': 'cuda',
}


def write_once(path, value):
    data = (canonical(value) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)


def begin_execution(
    lab, attempt_id, task_id, model_id, *, generation=None, arm='frozen', memory_policy='none'
):
    identifier(attempt_id)
    identifier(arm)
    fixed_model = lab.db.execute("SELECT value FROM metadata WHERE key='study_model'").fetchone()
    if fixed_model and fixed_model['value'] != model_id:
        raise LabError('试点实际模型与预定模型不一致')
    model = lab.model(model_id)
    if model['evidence_level'] != 'model_smoke' or not model.get('base_identity'):
        raise LabError('真实入口只接纳已登记的真实模型资格工件')
    spec = dict(DEFAULT_GENERATION if generation is None else generation)
    if (
        set(spec) != set(DEFAULT_GENERATION)
        or spec['do_sample'] is not False
        or spec['device'] not in {'cpu', 'cuda'}
    ):
        raise LabError('只接受冻结的贪心生成合同')
    for key, limit in [('max_new_tokens', 512), ('max_input_tokens', 4096), ('seed', 2**31 - 1)]:
        if type(spec[key]) is not int or not 0 < spec[key] <= limit:
            raise LabError('生成参数越界')
    if lab.db.execute(
        'SELECT 1 FROM attempts WHERE id=? OR (task_id=? AND model=? AND arm=?)',
        (attempt_id, task_id, model_id, arm),
    ).fetchone():
        raise LabError('已有首次尝试，失败或未知执行不能重发')
    old_arm = lab.db.execute("SELECT value FROM metadata WHERE key='arm'").fetchone()
    if old_arm and old_arm['value'] != arm:
        raise LabError('比较臂不能切换')
    old_spec = lab.db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()
    if old_spec and old_spec['value'] != canonical(spec):
        raise LabError('同一工作区不得事后更换生成设置')
    old_memory = lab.db.execute("SELECT value FROM metadata WHERE key='memory_policy'").fetchone()
    if old_memory and old_memory['value'] != memory_policy:
        raise LabError('不能切换已冻结记忆策略')
    from .memory import build_context

    context = build_context(lab, task_id, memory_policy)
    task = lab.public_task(task_id)
    request = {
        'schema': 2,
        'producer': 'serving_lab.inference.v1',
        'attempt_id': attempt_id,
        'task': task,
        'context': context,
        'task_hash': digest(lab.task(task_id)),
        'model_id': model_id,
        'artifact_digest': model['artifact']['digest'],
        'parameter_identity': model['parameter_identity'],
        'arm': arm,
        'generation': spec,
        'created': time.time(),
    }
    directory = lab.root / 'executions' / attempt_id
    if directory.exists():
        raise LabError('执行目录已存在，不覆盖原证据')
    directory.mkdir(parents=True)
    request_hash = write_once(directory / 'request.json', request)
    with lab.db:
        lab.db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('arm', arm))
        lab.db.execute(
            'INSERT OR IGNORE INTO metadata VALUES(?,?)', ('generation', canonical(spec))
        )
        lab.db.execute(
            'INSERT OR IGNORE INTO metadata VALUES(?,?)', ('memory_policy', memory_policy)
        )
        lab.db.execute(
            'INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
            (
                attempt_id,
                task_id,
                model_id,
                arm,
                'local_hf',
                'null',
                time.time(),
                model['artifact']['digest'],
            ),
        )
        lab.db.execute(
            'INSERT INTO executions VALUES(?,?,?,?,?)',
            (attempt_id, 'reserved', request_hash, None, str(directory)),
        )
        lab.event('execution_reserved', {'attempt_id': attempt_id, 'request_hash': request_hash})
    return request


def save_input(lab, attempt_id, rendered, token_ids, software):
    row = lab.db.execute('SELECT * FROM executions WHERE attempt_id=?', (attempt_id,)).fetchone()
    if not row or row['state'] != 'reserved':
        raise LabError('执行不处于预约状态')
    directory = Path(row['directory'])
    if (
        not isinstance(rendered, str)
        or not token_ids
        or any(type(t) is not int or t < 0 for t in token_ids)
        or set(software.get('source_files', {}))
        != {
            'source_inference.py',
            'source_execution.py',
            'source_lab.py',
            'source_business.py',
            'source_memory.py',
        }
        or type(software.get('tokenizer_eos_token_id')) is not int
    ):
        raise LabError('实际输入或源码资格不完整')
    for name, expected in software['source_files'].items():
        if file_hash(directory / name) != expected:
            raise LabError('源码快照字节不匹配')
    payload = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    from .memory import input_messages

    value = {
        'messages': input_messages(payload),
        'rendered_prompt': rendered,
        'input_ids': token_ids,
        'software': software,
    }
    write_once(directory / 'input.json', value)
    with lab.db:
        lab.db.execute("UPDATE executions SET state='running' WHERE attempt_id=?", (attempt_id,))
        lab.event(
            'execution_started',
            {'attempt_id': attempt_id, 'input_sha256': file_hash(directory / 'input.json')},
        )


def finish_execution(lab, attempt_id, raw_text, output_ids, *, ended_eos, elapsed_seconds, runtime):
    row = lab.db.execute('SELECT * FROM executions WHERE attempt_id=?', (attempt_id,)).fetchone()
    if not row or row['state'] != 'running':
        raise LabError('执行不处于运行状态')
    directory = Path(row['directory'])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    model = lab.model(request['model_id'])
    if model['artifact']['digest'] != request['artifact_digest']:
        raise LabError('执行时模型身份漂移')
    actual_input = json.loads((directory / 'input.json').read_text(encoding='utf-8'))
    if (
        not isinstance(raw_text, str)
        or not output_ids
        or any(type(t) is not int or t < 0 for t in output_ids)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
        or ended_eos != (output_ids[-1] == actual_input['software']['tokenizer_eos_token_id'])
        or runtime.get('loaded_model_id') != request['model_id']
        or runtime.get('loaded_artifact_digest') != request['artifact_digest']
        or runtime.get('loaded_parameter_identity') != request['parameter_identity']
        or runtime.get('prompt_tokens') != len(actual_input['input_ids'])
        or runtime.get('completion_tokens') != len(output_ids)
    ):
        raise LabError('实际加载身份、token计量或结束原因不一致')
    try:
        output = json.loads(raw_text)
        canonical(output)
        json_valid = isinstance(output, dict)
    except (ValueError, TypeError):
        output = None
        json_valid = False
    response = {
        'raw_text': raw_text,
        'output_ids': output_ids,
        'parsed': output,
        'json_valid': json_valid,
        'output_complete': bool(ended_eos),
        'elapsed_seconds': elapsed_seconds,
        'runtime': runtime,
    }
    write_once(directory / 'response.json', response)
    receipt = {
        'producer': 'serving_lab.inference.v1',
        'evidence_level': 'model_smoke',
        'attempt_id': attempt_id,
        'request_hash': row['request_hash'],
        'input_hash': file_hash(directory / 'input.json'),
        'response_hash': file_hash(directory / 'response.json'),
        'artifact_digest': request['artifact_digest'],
        'parameter_identity': request['parameter_identity'],
        'state': 'completed',
        'completed_at': time.time(),
    }
    receipt_hash = write_once(directory / 'receipt.json', receipt)
    with lab.db:
        lab.db.execute('UPDATE attempts SET output=? WHERE id=?', (canonical(output), attempt_id))
        lab.db.execute(
            "UPDATE executions SET state='completed',receipt_hash=? WHERE attempt_id=?",
            (receipt_hash, attempt_id),
        )
        lab.event('execution_completed', {'attempt_id': attempt_id, 'receipt_hash': receipt_hash})
    return response


def fail_execution(lab, attempt_id, error):
    row = lab.db.execute('SELECT * FROM executions WHERE attempt_id=?', (attempt_id,)).fetchone()
    if not row or row['state'] not in {'reserved', 'running'}:
        return
    directory = Path(row['directory'])
    error_hash = write_once(
        directory / 'error.json',
        {'type': type(error).__name__, 'message': str(error), 'attempt_id': attempt_id},
    )
    with lab.db:
        lab.db.execute(
            "UPDATE executions SET state='failed',receipt_hash=? WHERE attempt_id=?",
            (error_hash, attempt_id),
        )
        lab.event('execution_failed', {'attempt_id': attempt_id, 'error_hash': error_hash})


def verify_execution(lab, attempt_id, *, _cache=None):
    # 只在一次核验调用内复用结果，避免经历链递归重复哈希同一历史。
    if _cache is None:
        _cache = {}
    if attempt_id in _cache:
        return _cache[attempt_id]
    attempt = lab.db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
    row = lab.db.execute('SELECT * FROM executions WHERE attempt_id=?', (attempt_id,)).fetchone()
    if not attempt or not row or attempt['actor'] != 'local_hf':
        raise LabError('缺少受控执行预约')
    directory = Path(row['directory'])
    if directory.resolve() != (lab.root / 'executions' / attempt_id).resolve():
        raise LabError('执行证据目录不匹配')
    if file_hash(directory / 'request.json') != row['request_hash']:
        raise LabError('请求证据被改写')
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    task = lab.task(attempt['task_id'])
    model = lab.model(attempt['model'])
    public = {key: task[key] for key in ('task_id', 'rule_version', 'messages')}
    if (
        request['task'] != public
        or request['task_hash'] != digest(task)
        or request['attempt_id'] != attempt_id
        or request['model_id'] != attempt['model']
        or request['arm'] != attempt['arm']
        or request['artifact_digest'] != model['artifact']['digest']
        or attempt['artifact_digest'] != model['artifact']['digest']
        or request['parameter_identity'] != model['parameter_identity']
    ):
        raise LabError('执行身份不匹配')
    result = {
        'state': row['state'],
        'request_hash': row['request_hash'],
        'receipt_hash': row['receipt_hash'],
        'complete': False,
        'output_complete': False,
        'json_valid': False,
    }
    if row['state'] == 'failed':
        if file_hash(directory / 'error.json') != row['receipt_hash']:
            raise LabError('失败回执被修改')
    if row['state'] != 'completed':
        return result
    if file_hash(directory / 'receipt.json') != row['receipt_hash']:
        raise LabError('回执被改写')
    receipt = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
    if (
        receipt['producer'] != 'serving_lab.inference.v1'
        or receipt['evidence_level'] != 'model_smoke'
        or receipt['state'] != 'completed'
        or receipt['request_hash'] != row['request_hash']
        or receipt['attempt_id'] != attempt_id
        or receipt['artifact_digest'] != model['artifact']['digest']
        or receipt['parameter_identity'] != model['parameter_identity']
    ):
        raise LabError('回执身份不匹配')
    for name, key in [('input.json', 'input_hash'), ('response.json', 'response_hash')]:
        if file_hash(directory / name) != receipt[key]:
            raise LabError('执行输入或响应被改写')
    actual_input = json.loads((directory / 'input.json').read_text(encoding='utf-8'))
    response = json.loads((directory / 'response.json').read_text(encoding='utf-8'))
    from .memory import input_messages, verify_context

    if request.get('schema') == 2:
        verify_context(lab, request, _cache)
    if actual_input['messages'] != input_messages(request):
        raise LabError('实际模型消息不匹配')
    sources = {'source_inference.py', 'source_execution.py', 'source_lab.py'}
    if request.get('schema') == 2:
        sources |= {'source_business.py', 'source_memory.py'}
    if set(actual_input['software']['source_files']) != sources:
        raise LabError('缺少执行源码快照')
    for source, expected in actual_input['software']['source_files'].items():
        if Path(source).name != source or file_hash(directory / source) != expected:
            raise LabError('执行源码快照不一致')
    if canonical(response['parsed']) != attempt['output']:
        raise LabError('输出与账本不匹配')
    try:
        parsed = json.loads(response['raw_text'])
        canonical(parsed)
        valid = isinstance(parsed, dict)
    except (ValueError, TypeError):
        parsed = None
        valid = False
    if canonical(parsed) != attempt['output'] or valid != response['json_valid']:
        raise LabError('原始输出解析不匹配')
    runtime = response['runtime']
    if (
        runtime['loaded_model_id'] != request['model_id']
        or runtime['loaded_artifact_digest'] != request['artifact_digest']
        or runtime['loaded_parameter_identity'] != request['parameter_identity']
        or runtime['prompt_tokens'] != len(actual_input['input_ids'])
        or runtime['completion_tokens'] != len(response['output_ids'])
        or not response['output_ids']
        or response['output_complete']
        != (response['output_ids'][-1] == actual_input['software']['tokenizer_eos_token_id'])
    ):
        raise LabError('回执运行身份与实际输出计量不符')
    result.update(complete=True, output_complete=response['output_complete'], json_valid=valid)
    _cache[attempt_id] = result
    return result


def resume_existing(lab, task_id, model_id, generation, arm, memory_policy='none'):
    """显式继续只跳过已有首尝，不复做成功、失败或未知调用。"""
    row = lab.db.execute(
        'SELECT * FROM attempts WHERE task_id=? AND model=? AND arm=? ORDER BY created,id LIMIT 1',
        (task_id, model_id, arm),
    ).fetchone()
    if not row:
        return None
    evidence = verify_execution(lab, row['id'])
    entry = lab.db.execute(
        'SELECT directory FROM executions WHERE attempt_id=?', (row['id'],)
    ).fetchone()
    request = json.loads((Path(entry['directory']) / 'request.json').read_text(encoding='utf-8'))
    if request['generation'] != generation:
        raise LabError('继续执行不能更换已冻结生成设置')
    if request.get('context', {}).get('policy', 'none') != memory_policy:
        raise LabError('继续执行不能更换记忆策略')
    task = lab.task(task_id)
    if task['split'] == 'train' and evidence['complete']:
        old = lab.db.execute(
            'SELECT body FROM feedback WHERE attempt_id=?', (row['id'],)
        ).fetchone()
        if not old:
            # 完成回执与反馈之间的崩溃只补反馈，不重新生成；可用时点取当前曝光序号。
            lab.feedback(row['id'])
        else:
            verdict = json.loads(old['body'])
            expected = check_output(task, json.loads(row['output']))
            expected['passed'] = (
                expected['passed'] and evidence['output_complete'] and evidence['json_valid']
            )
            if {k: v for k, v in verdict.items() if k != 'available_sequence'} != expected:
                raise LabError('恢复时已存反馈与原始输出不符')
    return {
        'attempt_id': row['id'],
        'state': evidence['state'],
        'reused': True,
        'new_model_call': False,
    }
