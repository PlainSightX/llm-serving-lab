import json
import subprocess
import sys


def test_real_cli_demo_and_reload(tmp_path):
    out = tmp_path / 'run'
    proc = subprocess.run(
        [sys.executable, '-m', 'serving_lab.cli', 'demo', '--output', str(out)],
        capture_output=True,
        encoding='utf-8',
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result['checks']['bad_candidate_rejected']
    assert result['checks']['rollback_verified']
    assert result['checks']['training_examples'] == 8
    second = subprocess.run(
        [sys.executable, '-m', 'serving_lab.cli', '--workspace', str(out / 'workspace'), 'status'],
        capture_output=True,
        encoding='utf-8',
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)['active'] == 'fixture_v1'
    assert (out / 'report.html').is_file()


def test_report_escapes_html(tmp_path):
    from serving_lab.reporting import write_report

    p = write_report({'active': '<script>danger</script>'}, tmp_path)
    assert '<script>' not in p.read_text(encoding='utf-8')
