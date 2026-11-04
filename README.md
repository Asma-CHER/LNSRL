# LNS-RL-KES
 
## Paper checkpoint

The bundled model is the checkpoint loaded by the paper evaluator:

```text
models/lns_rl_kes.pth
SHA-256: ffcf8f83546a665c3cecd8d0b0a904a9d0ce3854a75dd12636fb3239b3efb664 
 
```

It is the 11-input-feature LNS-RL-V1 policy. The final four inputs are the
route geometry metrics; 

Run the release integrity check at any time:

```bash
python verify_release.py
```

## Requirements

- Python 3.10 or newer
- Java JDK 11 or newer (`java` and `javac` on `PATH`)
- PyTorch and PyTorch Geometric compatible with the selected CPU/CUDA build

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Solve a CVRPLIB instance

The defaults reproduce the paper search configuration: 5,000 LNS iterations,
two anchors, initial temperature 100, and final temperature 1.

```bash
python solve.py \
  --instance data/paper_instances/CMT1.vrp \
  --solution data/paper_solutions/CMT1.sol \
  --device cpu \
  --output results/generated/CMT1.json
```

Reduce only the iteration count:

```bash
python solve.py \
  --instance data/paper_instances/CMT1.vrp \
  --solution data/paper_solutions/CMT1.sol \
  --iterations 5 \
  --device cpu \
  --output results/generated/CMT1.json
```

For another CVRPLIB instance:

```bash
python solve.py --instance /path/to/instance.vrp --iterations 5000
```

The `.sol` argument is optional. When supplied, the JSON output includes the
reference cost and percentage gap. As in the paper evaluator, this release
accepts CVRP instances with fewer than 200 customers.

## Exact paper protocol

The historical evaluation used:

- 41 CVRP benchmarks: CMT, tai, and X instances in `data/`;
- 10 independent executions per instance;
- 5,000 search iterations per execution;
- two learned anchors;
- 11 node/route features;
- simulated annealing from 100 to 1;
- the checkpoint identified above.

The Java initializer uses its own random generator, so `--seed` controls NumPy
and PyTorch but does not make the entire Java/Python run bitwise deterministic.
Use independent repeated executions when regenerating aggregate results.

## Published results

`results/Results_KES2025.xlsx` is the unchanged final workbook used to prepare
the paper tables.
 

The four workbook sheets are also exported as CSV:

- `kes2025_all.csv`: all best, average, and CPU columns;
- `paper_best.csv`: best solution table;
- `paper_average.csv`: average solution table;
- `paper_average_cpu.csv`: average execution-time table.

 

## Layout

```text
solve.py                 paper-faithful single-instance runner
verify_release.py        checkpoint, workbook, and dataset integrity check
lib/                     LNS-RL-V1 policy and environment
JAVA/                    initialization, SISR baseline, and regret repair
models/lns_rl_kes.pth    exact paper checkpoint
data/paper_instances/    41 instances reported in the KES workbook
data/paper_solutions/    corresponding reference solutions
results/                 final workbook and CSV table exports
```
 