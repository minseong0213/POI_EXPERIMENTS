"""Continue submitted GPU work through delivery; journal every completed step."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORCH = Path('/home/mlops/gpu-orchestrator')
STUDY = ROOT / 'artifacts/poi-adversarial-20260915-001'
RCLONE = ['rclone', '--config', str(ORCH / 'secrets/rclone.conf')]


def command(args, timeout=600):
    return subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def checked(args, timeout=600, attempts=3):
    for attempt in range(attempts):
        result = command(args, timeout)
        if result.returncode == 0:
            return result.stdout
        if attempt + 1 < attempts:
            time.sleep(10)
    raise RuntimeError(f'{Path(str(args[0])).name} failed: {result.stdout[-2500:]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection', required=True)
    parser.add_argument('--evaluation', default='poi-attack-evaluation-20260915-001')
    parser.add_argument('--explanation', default='poi-selected-explain-20260915-001')
    args = parser.parse_args()
    connection = {}
    for line in Path(args.connection).read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            connection[key] = value.strip('\"\'')
    host, port = connection['HOST'], connection['PORT']
    state_dir = STUDY / 'workflow'
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'manager.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = state_dir / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed': []}

    def record(phase, **values):
        state.update(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(state_path)
        print(state['updated_at'], phase, flush=True)

    def step(name, action):
        if name in state['completed']:
            return
        record(name)
        action()
        state['completed'].append(name)
        record(name + '_complete')

    def wait_run(run):
        deadline = time.monotonic() + 11 * 3600
        failures = 0
        while time.monotonic() < deadline:
            result = command([ORCH / 'scripts/status_experiment.sh', host, port, 'poi', run], 90)
            if 'RUN FINISHED: exit 0' in result.stdout:
                return
            if 'RUN FINISHED: exit ' in result.stdout:
                checked([ORCH / 'scripts/sync_results.sh', host, port, 'poi', run], 1800)
                raise RuntimeError(f'{run} failed: {result.stdout[-2500:]}')
            if 'RUN ACTIVE' in result.stdout:
                failures = 0
                progress = [line for line in result.stdout.splitlines() if line.startswith('completed ')]
                record('running_' + run, progress=progress[-1:] or state.get('progress', []))
            else:
                failures += 1
                if failures >= 10:
                    raise RuntimeError(f'Cannot verify worker {run}: {result.stdout[-1000:]}')
            time.sleep(45)
        raise RuntimeError(f'Wait deadline exceeded: {run}')

    def sync(run):
        checked([ORCH / 'scripts/sync_results.sh', host, port, 'poi', run], 1800)

    def download(run, stage):
        checked(RCLONE + ['copy', f'r2:ml-experiments/results/poi/{run}/reports/{stage}',
                          str(STUDY / 'reports' / stage), '--exclude', '/work/**'], 1800)

    def submit_explanation():
        # Check before submit; an SSH error after submission must never duplicate a job.
        result = command([ORCH / 'scripts/status_experiment.sh', host, port, 'poi', args.explanation], 90)
        if 'RUN ACTIVE' in result.stdout or 'RUN FINISHED:' in result.stdout:
            return
        if 'RUN UNKNOWN/INTERRUPTED' not in result.stdout:
            raise RuntimeError('Cannot establish explanation job state before submission')
        script = '''set -Eeuo pipefail
source "$1"
source "$2"
DATA_DIR=/workspace/data/poi-34k RUN_DETACH=1 RUN_MAX_SECONDS=14400 /home/mlops/gpu-orchestrator/scripts/run_experiment.sh "$HOST" "$PORT" poi "$3" -- "${COMMAND[@]}"
'''
        result = command(['bash', '-c', script, 'submit', str(ROOT / 'experiments/selected-explain-tabpfn.env'),
                          args.connection, args.explanation], 120)
        if result.returncode:
            verify = command([ORCH / 'scripts/status_experiment.sh', host, port, 'poi', args.explanation], 90)
            if 'RUN ACTIVE' not in verify.stdout and 'RUN FINISHED:' not in verify.stdout:
                raise RuntimeError('Explanation submission was not confirmed')

    try:
        step('evaluation', lambda: wait_run(args.evaluation))
        step('evaluation_sync', lambda: sync(args.evaluation))
        step('evaluation_download', lambda: download(args.evaluation, '06_attack_evaluation'))
        step('explanation_submit', submit_explanation)
        step('explanation', lambda: wait_run(args.explanation))
        step('explanation_sync', lambda: sync(args.explanation))
        step('explanation_download', lambda: download(args.explanation, '07_explainability'))
        step('destroy_gpu', lambda: checked([ORCH / 'scripts/vast/destroy_gpu.sh', connection['INSTANCE_ID'],
            '--yes', ORCH / 'config/vast.env'], 1800, attempts=1))
        step('audit', lambda: checked([sys.executable, ROOT / 'scripts/audit_attack_scope.py']))
        step('delivery', lambda: checked([sys.executable, ROOT / 'scripts/bundle_attack_results.py']))
        step('publish', lambda: checked(RCLONE + ['copy', str(STUDY / 'reports'),
            'r2:ml-experiments/results/poi/poi-adversarial-20260915-001/reports', '--exclude', '**/work/**'], 1800))
        record('complete', status='complete')
    except Exception as error:
        record('failed', status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
