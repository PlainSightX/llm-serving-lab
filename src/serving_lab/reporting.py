"""报告仅解释已保存的证据层级，不把控制面检查称为学习效果。"""

import html
import json
from pathlib import Path


def write_report(summary, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (directory / 'report.json').write_text(text + '\n', encoding='utf-8')
    rows = []
    for result in summary.get('evaluations', []):
        outcomes = result.get('outcomes', [])
        values = [
            result['model_id'],
            result['evidence_level'],
            f"{sum(bool(x['passed']) for x in outcomes)}/{len(outcomes)}",
            '通过' if result['passed'] else '未通过',
        ]
        rows.append(
            '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
        )
    table = (
        '<table><thead><tr><th>模型版本</th><th>证据层级</th><th>面板检查</th><th>采用门</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
        if rows
        else '<p>尚无评价记录。</p>'
    )
    page = (
        '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>经验巩固工作台 · 运行记录</title>
<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172f37;background:#f4f7f8}h1{font-size:28px}article{background:white;border:1px solid #dbe5e8;padding:24px;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace}strong{color:#974519}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe5e8}summary{cursor:pointer}</style>
<h1>经验巩固与受控模型更新</h1><article><p><strong>流程与小模型资格检查不等于持续学习效果。</strong></p>
<p>当前活动指针：<b>'''
        + html.escape(str(summary['active']))
        + '''</b></p>
<p>control_plane 为流程夹具；model_smoke 为真实模型资格。活动指针不表示远端服务或集群已经切换。</p>
'''
        + table
        + '''<details><summary>查看完整记录、来源与边界</summary><pre>'''
        + html.escape(text)
        + '''</pre></details></article></html>'''
    )
    (directory / 'report.html').write_text(page, encoding='utf-8')
    return directory / 'report.html'
