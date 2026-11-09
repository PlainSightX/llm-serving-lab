"""单写者研究账本：冻结输入、先执行后反馈、隔离训练与评价。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import time
from pathlib import Path

SOURCE_BYTES = Path(__file__).read_bytes()


class LabError(ValueError):
    """调用违反研究合同；不自动补造证据。"""


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def base_paths(directory):
    """基座是不可变目录，包含嵌套chat_templates等加载输入；不接受链接逃逸。"""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise LabError('基座目录不存在')
    paths = {}
    for path in directory.rglob('*'):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise LabError('基座禁止符号链接或目录连接')
        if path.is_file():
            paths[path.relative_to(directory).as_posix()] = path
    return paths


def inspect_base(directory):
    directory = Path(directory).resolve()
    members = {}
    for name, path in base_paths(directory).items():
        members[name] = file_hash(path)
    result = {'path': str(directory), 'files': members}
    Lab.verify_base(result)
    return result


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise LabError('标识必须是1至100位字母、数字、下划线或短横线')
    return value


def validate_task(task):
    required = {'task_id', 'sequence', 'family', 'split', 'rule_version', 'messages', 'expected'}
    if set(task) not in (required, required | {'evaluation'}):
        raise LabError(f'任务字段必须恰为 {sorted(required)}')
    identifier(task['task_id'])
    identifier(task['family'])
    identifier(task['rule_version'])
    if type(task['sequence']) is not int or task['sequence'] < 0:
        raise LabError('sequence必须为非负整数')
    if task['split'] not in {'train', 'dev', 'holdout'}:
        raise LabError('未知切分')
    if not isinstance(task['expected'], dict) or not task['expected']:
        raise LabError('expected必须为非空JSON对象')
    messages = task['messages']
    if not isinstance(messages, list) or not messages:
        raise LabError('messages必须为非空列表')
    for m in messages:
        if (
            set(m) != {'role', 'content'}
            or m['role'] not in {'system', 'user'}
            or not isinstance(m['content'], str)
        ):
            raise LabError('任务公开输入只接受system/user文本消息')
    if not any(m['role'] == 'user' for m in messages):
        raise LabError('任务必须包含user消息')
    canonical(task)
    if 'evaluation' in task:
        from .business import validate_business_task

        validate_business_task(task)
    return task


def check_output(task, output):
    """一期采用可审计的精确结构比较；不把它称通用语义裁判。"""
    if 'evaluation' in task:
        from .business import verify_plan

        return {
            **verify_plan(task['evaluation']['case'], output),
            'verifier': task['evaluation']['kind'],
            'task_hash': digest(task),
            'output_hash': digest(output),
        }
    return {
        'passed': canonical(output) == canonical(task['expected']),
        'verifier': 'exact-json-v1',
        'task_hash': digest(task),
        'output_hash': digest(output),
    }


def inspect_adapter(directory):
    """检查safetensors布局与所有工件摘要；可装载性另由训练回执证明。"""
    directory = Path(directory).resolve()
    config_path = directory / 'adapter_config.json'
    weights = directory / 'adapter_model.safetensors'
    if not config_path.is_file() or not weights.is_file():
        raise LabError('缺少adapter配置或参数文件')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('peft_type') != 'LORA' or not config.get('base_model_name_or_path'):
        raise LabError('不是有明确基座的LoRA')
    size = weights.stat().st_size
    with weights.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise LabError('权重头不完整')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(size - 8, 16 * 1024 * 1024):
            raise LabError('权重头长度无效')
        try:
            header = json.loads(f.read(length))
        except Exception as exc:
            raise LabError('无效权重JSON头') from exc
    tensors = [v for k, v in header.items() if k != '__metadata__']
    dtype_bytes = {
        'F32': 4,
        'F16': 2,
        'BF16': 2,
        'F64': 8,
        'I64': 8,
        'I32': 4,
        'I16': 2,
        'I8': 1,
        'U8': 1,
        'BOOL': 1,
    }
    ranges = []
    for tensor in tensors:
        try:
            count = 1
            for dim in tensor['shape']:
                if type(dim) is not int or dim < 0:
                    raise ValueError()
                count *= dim
            a, b = tensor['data_offsets']
            if (
                type(a) is not int
                or type(b) is not int
                or a < 0
                or b - a != count * dtype_bytes[tensor['dtype']]
            ):
                raise ValueError()
            ranges.append((a, b))
        except Exception as exc:
            raise LabError('参数张量形状/字节范围无效') from exc
    ranges.sort()
    cursor = 0
    for a, b in ranges:
        if a != cursor:
            raise LabError('参数数据不连续或重叠')
        cursor = b
    if not tensors or cursor <= 0 or cursor != size - 8 - length:
        raise LabError('参数字节数量不一致')
    members = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise LabError('模型工件禁止符号链接')
        if p.is_file():
            members[p.relative_to(directory).as_posix()] = file_hash(p)
    return {
        'path': str(directory),
        'files': members,
        'digest': digest(members),
        'base': config['base_model_name_or_path'],
        'tensor_count': len(tensors),
    }


class Lab:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'writer.lock').open('a+b')
        self.lock.seek(0, 2)
        if self.lock.tell() == 0:
            self.lock.write(b'0')
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise LabError('工作区已有写入者') from exc
        self.db = sqlite3.connect(self.root / 'lab.sqlite')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, seq INTEGER UNIQUE, split TEXT, family TEXT, hash TEXT, body TEXT);
          CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, task_id TEXT, model TEXT, arm TEXT, actor TEXT, output TEXT, created REAL, artifact_digest TEXT);
          CREATE TABLE IF NOT EXISTS feedback(attempt_id TEXT PRIMARY KEY, body TEXT, created REAL);
          CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, body TEXT);
          CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY, body TEXT);
          CREATE TABLE IF NOT EXISTS releases(id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, evaluation TEXT, previous TEXT, created REAL);
          CREATE TABLE IF NOT EXISTS active(slot INTEGER PRIMARY KEY CHECK(slot=1), model TEXT);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, body TEXT, created REAL);
          CREATE TABLE IF NOT EXISTS exposures(task_id TEXT PRIMARY KEY, seq INTEGER, created REAL);
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE IF NOT EXISTS executions(attempt_id TEXT PRIMARY KEY, state TEXT, request_hash TEXT, receipt_hash TEXT, directory TEXT);
        ''')
        # 旧控制面演示可读；没有旧工件绑定的尝试永远不能晋级为新候选证据。
        if 'artifact_digest' not in {r[1] for r in self.db.execute('PRAGMA table_info(attempts)')}:
            self.db.execute('ALTER TABLE attempts ADD COLUMN artifact_digest TEXT')
        self.db.commit()
        # 写入锁已排除存活的同工作区执行者；孤立预约保留为未知，不能重发洗掉首败。
        with self.db:
            interrupted = self.db.execute(
                "SELECT attempt_id FROM executions WHERE state IN ('reserved','running')"
            ).fetchall()
            self.db.execute(
                "UPDATE executions SET state='unknown' WHERE state IN ('reserved','running')"
            )
            for row in interrupted:
                self.event('execution_interrupted', {'attempt_id': row['attempt_id']})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()
        self.lock.close()  # 操作系统锁随句柄关闭释放；崩溃不依赖删除PID文件。

    def event(self, kind, body):
        self.db.execute(
            'INSERT INTO events(kind,body,created) VALUES(?,?,?)',
            (kind, canonical(body), time.time()),
        )

    def import_tasks(self, tasks):
        if not tasks:
            raise LabError('空任务集')
        if (
            self.db.execute('SELECT count(*) FROM exposures').fetchone()[0]
            or self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]
        ):
            raise LabError('任务流已开始，输入与保护面板不可再改')
        with self.db:
            for task in tasks:
                validate_task(task)
                old = self.db.execute(
                    'SELECT hash FROM tasks WHERE id=?', (task['task_id'],)
                ).fetchone()
                if old:
                    if old['hash'] != digest(task):
                        raise LabError('已有任务不可改写')
                    continue
                for row in self.db.execute('SELECT split,family,body FROM tasks'):
                    previous = json.loads(row['body'])
                    if row['split'] != task['split'] and (
                        row['family'] == task['family'] or previous['messages'] == task['messages']
                    ):
                        raise LabError('跨切分的任务族/完全相同输入泄漏')
                try:
                    self.db.execute(
                        'INSERT INTO tasks VALUES(?,?,?,?,?,?)',
                        (
                            task['task_id'],
                            task['sequence'],
                            task['split'],
                            task['family'],
                            digest(task),
                            canonical(task),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise LabError('任务ID或序号冲突') from exc
            self.event('import_tasks', {'ids': [t['task_id'] for t in tasks]})

    def task(self, task_id):
        row = self.db.execute('SELECT body,hash FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            raise LabError('任务不存在')
        task = json.loads(row['body'])
        if digest(task) != row['hash']:
            raise LabError('任务字节身份变更')
        return task

    def public_task(self, task_id):
        task = self.task(task_id)
        if not self.db.execute('SELECT 1 FROM exposures WHERE task_id=?', (task_id,)).fetchone():
            next_task = self.db.execute(
                'SELECT id FROM tasks WHERE id NOT IN (SELECT task_id FROM exposures) ORDER BY seq LIMIT 1'
            ).fetchone()
            if not next_task or next_task['id'] != task_id:
                raise LabError('禁止越序曝光未来任务')
            with self.db:
                self.db.execute(
                    'INSERT INTO exposures VALUES(?,?,?)', (task_id, task['sequence'], time.time())
                )
                self.event('expose', {'task_id': task_id, 'sequence': task['sequence']})
        return {k: task[k] for k in ('task_id', 'rule_version', 'messages')}

    def record(self, attempt_id, task_id, model, output, arm='frozen', actor='scripted'):
        identifier(attempt_id)
        identifier(model)
        identifier(arm)
        if actor not in {'scripted', 'manual'}:
            raise LabError('手工入口不能声明真实模型执行；需要独立受控推理入口')
        self.public_task(task_id)
        current_arm = self.db.execute('SELECT value FROM metadata WHERE key=?', ('arm',)).fetchone()
        if current_arm and current_arm['value'] != arm:
            raise LabError('每个工作区只能承载一个比较臂，禁止换名重跑')
        row = self.db.execute('SELECT body FROM models WHERE id=?', (model,)).fetchone()
        artifact_digest = self.model(model)['artifact']['digest'] if row else None
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('arm', arm))
            try:
                self.db.execute(
                    'INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
                    (
                        attempt_id,
                        task_id,
                        model,
                        arm,
                        actor,
                        canonical(output),
                        time.time(),
                        artifact_digest,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LabError('尝试身份已存在，禁止覆盖原输出') from exc
            self.event(
                'attempt',
                {'attempt_id': attempt_id, 'task_id': task_id, 'model': model, 'actor': actor},
            )

    def feedback(self, attempt_id):
        row = self.db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
        if not row:
            raise LabError('先执行并保存结果，再开放反馈')
        if row['actor'] == 'local_hf':
            from .execution import verify_execution

            result = verify_execution(self, attempt_id)
            if not result['complete']:
                raise LabError('未知/失败执行不得产生可学习反馈')
        verdict = check_output(self.task(row['task_id']), json.loads(row['output']))
        if row['actor'] == 'local_hf':
            verdict['passed'] = (
                verdict['passed'] and result['output_complete'] and result['json_valid']
            )
        latest = self.db.execute('SELECT max(seq) FROM exposures').fetchone()[0]
        verdict['available_sequence'] = latest
        with self.db:
            try:
                self.db.execute(
                    'INSERT INTO feedback VALUES(?,?,?)',
                    (attempt_id, canonical(verdict), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise LabError('反馈不可覆盖') from exc
            self.event('feedback', {'attempt_id': attempt_id, 'passed': verdict['passed']})
        return verdict

    def export_training(self, destination, *, cutoff, arm='frozen'):
        if type(cutoff) is not int or cutoff < 0:
            raise LabError('无效截止序号')
        rows = self.db.execute(
            '''SELECT a.*,t.body task,t.hash task_hash,t.seq,f.body verdict
          FROM attempts a JOIN tasks t ON t.id=a.task_id JOIN feedback f ON f.attempt_id=a.id
          WHERE t.split='train' AND t.seq<=? AND a.arm=? ORDER BY t.seq,a.created''',
            (cutoff, arm),
        ).fetchall()
        selected = {}
        for row in rows:
            task = json.loads(row['task'])
            output = json.loads(row['output'])
            verdict = json.loads(row['verdict'])
            core = {k: v for k, v in verdict.items() if k != 'available_sequence'}
            checked = check_output(task, output)
            if row['actor'] == 'local_hf':
                from .execution import verify_execution

                observed = verify_execution(self, row['id'])
                checked['passed'] = checked['passed'] and all(
                    observed[k] for k in ('complete', 'output_complete', 'json_valid')
                )
            if digest(task) != row['task_hash'] or core != checked:
                raise LabError('经验来源被改写')
            if (
                verdict['passed']
                and verdict['available_sequence'] <= cutoff
                and row['task_id'] not in selected
            ):
                selected[row['task_id']] = row
        if not selected:
            raise LabError('没有合格的过去训练经验')
        destination = Path(destination)
        if destination.exists():
            raise LabError('导出路径已存在，必须保留旧快照')
        destination.mkdir(parents=True)
        examples = []
        for row in selected.values():
            task = json.loads(row['task'])
            examples.append(
                {
                    'task_id': task['task_id'],
                    'messages': task['messages']
                    + [{'role': 'assistant', 'content': row['output']}],
                    'source_attempt': row['id'],
                    'task_hash': row['task_hash'],
                }
            )
        data = ''.join(canonical(e) + '\n' for e in examples)
        data_path = destination / 'train.jsonl'
        data_path.write_text(data, encoding='utf-8')
        level = (
            'model_smoke'
            if all(r['actor'] == 'local_hf' for r in selected.values())
            else 'control_plane'
        )
        manifest = {
            'schema': 1,
            'cutoff': cutoff,
            'arm': arm,
            'task_ids': list(selected),
            'count': len(examples),
            'data_sha256': file_hash(data_path),
            'evidence_level': level,
            'created': time.time(),
        }
        (destination / 'manifest.json').write_text(canonical(manifest) + '\n', encoding='utf-8')
        with self.db:
            self.event('training_export', manifest)
        return manifest

    def register(self, model_id, directory, *, fixture=False):
        identifier(model_id)
        artifact = inspect_adapter(directory)
        receipt_path = Path(directory) / 'training-receipt.json'
        if fixture:
            level = 'control_plane'
            base_identity = None
        else:
            if not receipt_path.is_file():
                raise LabError('缺少实际训练/装载回执')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if (
                receipt.get('producer') != 'serving_lab.train.v1'
                or receipt.get('steps', 0) <= 0
                or not receipt.get('reload_verified')
            ):
                raise LabError('训练未完成或未验证重载')
            if receipt.get('adapter_sha256') != artifact['files']['adapter_model.safetensors']:
                raise LabError('参数与训练回执不符')
            if receipt.get('before_sha256') == receipt.get('adapter_sha256'):
                raise LabError('参数未改变')
            if Path(receipt.get('base_path', '')).resolve() != Path(artifact['base']).resolve():
                raise LabError('配置和训练回执的基座不一致')
            base_identity = {'path': receipt['base_path'], 'files': receipt.get('base_files', {})}
            self.verify_base(base_identity)
            level = receipt.get('purpose')
            if level not in {'model_smoke', 'research'}:
                raise LabError('未知训练证据层级')
        config = json.loads((Path(directory) / 'adapter_config.json').read_text(encoding='utf-8'))
        config.pop('base_model_name_or_path', None)
        parameter_identity = digest(
            {
                'weights': artifact['files']['adapter_model.safetensors'],
                'config': config,
                'base': base_identity['files'] if base_identity else artifact['base'],
            }
        )
        for row in self.db.execute('SELECT body FROM models'):
            previous = json.loads(row['body'])
            if (
                previous.get('parameter_identity') == parameter_identity
                or previous['artifact']['digest'] == artifact['digest']
            ):
                raise LabError('相同参数不可换名称重复登记或重跑保护面板')
        record = {
            'model_id': model_id,
            'artifact': artifact,
            'evidence_level': level,
            'base_identity': base_identity,
            'parameter_identity': parameter_identity,
        }
        with self.db:
            try:
                self.db.execute('INSERT INTO models VALUES(?,?)', (model_id, canonical(record)))
            except sqlite3.IntegrityError as exc:
                raise LabError('模型身份不可覆盖') from exc
            self.event('register', record)
        return record

    def register_base(self, model_id, directory):
        identifier(model_id)
        identity = inspect_base(directory)
        parameter_identity = digest({'base': identity['files']})
        for row in self.db.execute('SELECT body FROM models'):
            if json.loads(row['body']).get('parameter_identity') == parameter_identity:
                raise LabError('相同基座不可换名称重复登记')
        record = {
            'model_id': model_id,
            'kind': 'base',
            'base_identity': identity,
            'artifact': {
                'path': identity['path'],
                'files': identity['files'],
                'digest': digest(identity['files']),
                'base': identity['path'],
            },
            'parameter_identity': parameter_identity,
            'evidence_level': 'model_smoke',
        }
        with self.db:
            try:
                self.db.execute('INSERT INTO models VALUES(?,?)', (model_id, canonical(record)))
            except sqlite3.IntegrityError as exc:
                raise LabError('模型身份不可覆盖') from exc
            self.event('register_base', record)
        return record

    def model(self, model_id):
        row = self.db.execute('SELECT body FROM models WHERE id=?', (model_id,)).fetchone()
        if not row:
            raise LabError('模型未登记')
        result = json.loads(row['body'])
        if (
            result.get('kind') != 'base'
            and inspect_adapter(result['artifact']['path'])['digest']
            != result['artifact']['digest']
        ):
            raise LabError('登记后模型工件发生变化')
        if result.get('base_identity'):
            self.verify_base(result['base_identity'])
        return result

    @staticmethod
    def verify_base(identity):
        if (
            not identity['files']
            or 'config.json' not in identity['files']
            or not any(n.endswith('.safetensors') for n in identity['files'])
        ):
            raise LabError('基座清单不完整')
        directory = Path(identity['path'])
        if not directory.is_dir() or directory.is_symlink():
            raise LabError('基座目录不可用')
        paths = base_paths(directory)
        actual = set(paths)
        if actual != set(identity['files']):
            raise LabError('基座加载文件集合发生增减')
        for name, expected in identity['files'].items():
            path = paths[name]
            if not path.is_file() or path.is_symlink() or file_hash(path) != expected:
                raise LabError('基座或tokenizer内容发生变化')

    def evaluate(self, evaluation_id, model_id, task_ids, *, arm='frozen'):
        identifier(evaluation_id)
        model = self.model(model_id)
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise LabError('评价面板为空或重复')
        complete_panel = {
            r['id'] for r in self.db.execute("SELECT id FROM tasks WHERE split='holdout'")
        }
        if set(task_ids) != complete_panel:
            raise LabError('必须评价冻结任务流中的完整保护面板')
        current_arm = self.db.execute("SELECT value FROM metadata WHERE key='arm'").fetchone()
        if not current_arm or current_arm['value'] != arm:
            raise LabError('评价臂不匹配')
        outcomes = []
        all_model_executions = True
        for task_id in task_ids:
            task = self.task(task_id)
            if task['split'] != 'holdout':
                raise LabError('发布面板只接受holdout任务')
            # 首次尝试是评价对象，不能挑选重跑中的成功答案。
            row = self.db.execute(
                'SELECT * FROM attempts WHERE task_id=? AND model=? AND arm=? ORDER BY created,id LIMIT 1',
                (task_id, model_id, arm),
            ).fetchone()
            if not row:
                raise LabError('候选缺少面板任务的实际输出')
            if row['artifact_digest'] != model['artifact']['digest']:
                raise LabError('尝试未绑定当前模型工件；禁止事后补身份')
            check = check_output(task, json.loads(row['output']))
            execution = None
            if row['actor'] == 'local_hf':
                from .execution import verify_execution

                execution = verify_execution(self, row['id'])
                check['passed'] = (
                    check['passed']
                    and execution['complete']
                    and execution['output_complete']
                    and execution['json_valid']
                )
            else:
                all_model_executions = False
            outcomes.append(
                {
                    'task_id': task_id,
                    'attempt_id': row['id'],
                    'task_hash': digest(task),
                    **check,
                    'actor': row['actor'],
                    'execution': execution,
                }
            )
        result = {
            'evaluation_id': evaluation_id,
            'model_id': model_id,
            'artifact_digest': model['artifact']['digest'],
            'suite_digest': digest([(o['task_id'], o['task_hash']) for o in outcomes]),
            'arm': arm,
            'outcomes': outcomes,
            'passed': all(o['passed'] for o in outcomes),
            'evidence_level': 'model_smoke' if all_model_executions else 'control_plane',
        }
        with self.db:
            try:
                self.db.execute(
                    'INSERT INTO evaluations VALUES(?,?)', (evaluation_id, canonical(result))
                )
            except sqlite3.IntegrityError as exc:
                raise LabError('评价身份不可覆盖') from exc
            self.event('evaluate', result)
        return result

    def active(self):
        row = self.db.execute('SELECT model FROM active WHERE slot=1').fetchone()
        return row['model'] if row else None

    def promote(self, evaluation_id, *, expected_active=None, level='control_plane'):
        if level not in {'control_plane', 'model_smoke'}:
            raise LabError('当前只开放控制面/模型资格门；尚无研究与集群发布资格')
        row = self.db.execute(
            'SELECT body FROM evaluations WHERE id=?', (evaluation_id,)
        ).fetchone()
        if not row:
            raise LabError('评价不存在')
        evaluation = json.loads(row['body'])
        model = self.model(evaluation['model_id'])
        if (
            not evaluation['passed']
            or evaluation['evidence_level'] != level
            or model['evidence_level'] != level
        ):
            raise LabError('候选未通过同证据层级的采用门')
        if evaluation['artifact_digest'] != model['artifact']['digest']:
            raise LabError('工件不匹配')
        for outcome in evaluation['outcomes']:
            task = self.task(outcome['task_id'])
            attempt = self.db.execute(
                'SELECT * FROM attempts WHERE id=?', (outcome['attempt_id'],)
            ).fetchone()
            if (
                not attempt
                or attempt['model'] != model['model_id']
                or attempt['artifact_digest'] != model['artifact']['digest']
                or outcome['task_hash'] != digest(task)
                or outcome['output_hash'] != digest(json.loads(attempt['output']))
                or not check_output(task, json.loads(attempt['output']))['passed']
            ):
                raise LabError('评价来源发生变化')
            if level == 'model_smoke':
                from .execution import verify_execution

                observed = verify_execution(self, outcome['attempt_id'])
                if observed != outcome.get('execution') or not all(
                    observed[k] for k in ('complete', 'output_complete', 'json_valid')
                ):
                    raise LabError('实际执行回执已变化或未完成')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            previous = self.active()
            if previous != expected_active:
                raise LabError('活动版本已变化')
            self.db.execute(
                'INSERT INTO active VALUES(1,?) ON CONFLICT(slot) DO UPDATE SET model=excluded.model',
                (model['model_id'],),
            )
            self.db.execute(
                'INSERT INTO releases(model,evaluation,previous,created) VALUES(?,?,?,?)',
                (model['model_id'], evaluation_id, previous, time.time()),
            )
            self.event(
                'promote',
                {
                    'model_id': model['model_id'],
                    'evaluation_id': evaluation_id,
                    'previous': previous,
                    'scope': 'control_pointer_only',
                },
            )
        return model['model_id']

    def rollback(self, model_id, *, expected_active, level='control_plane'):
        row = self.db.execute(
            'SELECT evaluation FROM releases WHERE model=? ORDER BY id DESC LIMIT 1', (model_id,)
        ).fetchone()
        if not row:
            raise LabError('回滚目标从未通过发布门')
        return self.promote(row['evaluation'], expected_active=expected_active, level=level)

    def summary(self):
        return {
            'stage': 'local_control_plane',
            'active': self.active(),
            'counts': {
                table: self.db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                for table in (
                    'tasks',
                    'attempts',
                    'feedback',
                    'models',
                    'evaluations',
                    'releases',
                    'events',
                )
            },
            'evaluations': [
                json.loads(r['body']) for r in self.db.execute('SELECT body FROM evaluations')
            ],
            'models': [json.loads(r['body']) for r in self.db.execute('SELECT body FROM models')],
            'limitations': [
                '控制面发布指针不等于服务已切换',
                '协议夹具不能证明模型学习',
                '本地训练smoke不证明持续学习效果或集群能力',
            ],
        }
