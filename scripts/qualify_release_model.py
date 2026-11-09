"""先独立核验候选，失败不进入发布试验、不改变原质量面板。"""

import argparse
import asyncio
import json
from pathlib import Path

from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.execution import write_once
from serving_lab.parameters import bound_parameter_identity
from serving_lab.qualification import qualify_engine
from serving_lab.runtime import RuntimeStore
from start_engine import start, stop
from probe_serving import probe


async def run(config, output, panel_path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binding_path = output / 'worker/binding.json'
    engine = None
    try:
        start(config, output / 'worker')
        probe(binding_path, output / 'direct')
        binding = EngineBinding.load(binding_path)
        identity = bound_parameter_identity(binding)
        write_once(output / 'parameters.json', identity)
        engine = HttpEngine(binding)
        panel = json.loads(Path(panel_path).read_text(encoding='utf-8'))
        with RuntimeStore(output / 'runtime') as store:
            receipt = await qualify_engine(engine, panel, output / 'qualification', store=store)
        return receipt
    finally:
        if engine:
            await engine.close()
        if binding_path.exists():
            write_once(output / 'cleanup.json', stop(binding_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--panel', default='fixtures/serving/r2-quality.json')
    args = parser.parse_args()
    result = asyncio.run(run(args.config, args.output, args.panel))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
