"""Submit one self-contained LSF job per rank count; settings are in scripts."""
import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import verify_manifest


def job_script(root, config, ranks):
    command = shlex.join(["python3", str(root / "code/run_job.py"), "--ranks", str(ranks)])
    return f"""#!/bin/bash
#BSUB -J {config['name']}_p{ranks}
#BSUB -q hpcintro
#BSUB -n {config['slots']}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W {config['walltime_min']}
#BSUB -o {shlex.quote(str(root / f'jobs/p{ranks}/lsf_%J.out'))}
#BSUB -e {shlex.quote(str(root / f'jobs/p{ranks}/lsf_%J.err'))}
set -euo pipefail
cd {shlex.quote(str(root))}
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "job $LSB_JOBID; allocated slots $LSB_DJOB_NUMPROC; hosts $LSB_MCPU_HOSTS"
module list
lscpu
{command}
"""


def main():
    root = Path(__file__).resolve().parents[1]
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", nargs="+", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ranks = args.ranks or config["ranks"]
    if len(set(ranks)) != len(ranks) or any(r not in config["ranks"] for r in ranks):
        p.error("Choose unique rank counts from config.json")
    for count in ranks:
        script = job_script(root, config, count)
        if args.dry_run:
            print(script)
            continue
        folder = root / "jobs" / f"p{count}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "submitted.sh").write_text(script, encoding="utf-8", newline="\n")
        result = subprocess.run(["bsub"], input=script, text=True, capture_output=True)
        (folder / "submission.json").write_text(json.dumps({
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        }, indent=2))
        print(result.stdout, end="")
        print(result.stderr, end="")
        result.check_returncode()


if __name__ == "__main__":
    main()
