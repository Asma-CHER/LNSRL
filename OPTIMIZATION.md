# Runtime optimization

## Objective

This edition accelerates the KES 2025 LNS-RL implementation without changing
the trained checkpoint or the search algorithm. The original Java repair
method remains in `Recreate.step`; the optimization changes how Python invokes
that method.

## Original execution path

The original runner performed the following operations at every LNS iteration:

1. serialized the current routes and ruined customers to a temporary file;
2. started a new Java Virtual Machine;
3. reloaded the instance and distance matrix from disk;
4. ran `Recreate.step` once; and
5. parsed the process output in Python.

This process startup and repeated file I/O dominated the search time.

## Optimized execution path

`JAVA/RepairWorker.java` is started once at the beginning of a solve. It loads
the instance and distance matrix once, then accepts repair requests over a
buffered line protocol. Every request still calls the original
`Recreate.step` method.

Additional runtime-only improvements are:

- Java classes are recompiled only when a source is newer than its class file;
- the policy receives a PyTorch Geometric `Data` object directly, avoiding a
  one-element `DataLoader` on every iteration;
- the static nearest-neighbour edge tensor is constructed once; and
- PyTorch uses one CPU thread by default because the evaluated graphs are too
  small to benefit from multi-thread coordination.

The thread count can be changed when benchmarking a different machine:

```bash
python solve.py \
  --instance data/paper_instances/CMT1.vrp \
  --iterations 5000 \
  --torch-threads 1
```

## Benchmark

The following wall-clock measurements were collected on the same development
machine using CMT1, CPU inference, 100 iterations, and no animation:

| Runner | Time | Speedup |
| --- | ---: | ---: |
| Original `LNS-RL-KES` | 21.43 s | 1.00x |
| Optimized, first run including Java compilation | 7.52 s | 2.85x |
| Optimized, subsequent run | 5.81 s | 3.69x |

The benefit increases for long searches because the one-time model and process
startup costs are amortized over more iterations. Absolute timings depend on
the processor, Java runtime, PyTorch build, and instance size.

## Validation

The following checks were performed:

- the optimized policy call and the former one-item `DataLoader` call produced
  identical hidden states, selected nodes, and SISR coefficients for the same
  graph and PyTorch seed;
- the persistent worker and the original `Recreate` entry point produced an
  identical result for the same deterministic repair request;
- the bundled model retains SHA-256
  `ffcf8f83546a665c3cecd8d0b0a904a9d0ce3854a75dd12636fb3239b3efb664`;
- the published workbook retains SHA-256
  `309f6d6101b13bf5e7a7bebb870538c350fbd58ecf04b98338af50b48b3e2205`;
  and
- all 41 paper instance/solution pairs pass `verify_release.py`.

The Java repair operator uses an unseeded `Collections.shuffle`. Therefore,
the original runner was already nondeterministic, and separate executions are
not expected to follow bit-for-bit identical search trajectories. The
optimized worker uses the same shuffle and repair implementation.
