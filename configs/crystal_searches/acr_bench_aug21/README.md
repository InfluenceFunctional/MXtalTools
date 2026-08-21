# CPU vs GPU throughput benchmark — acridine sg14 Z′=2

```bash
python make_bench_configs.py
sbatch submit_bench_gpu.sbatch     # array 0-9
sbatch submit_bench_cpu.sbatch     # array 0-9
```

20 arms, 10 per device. Each device runs at **its own sensible operating point**
rather than on identical configs — matching them would answer a different
question (how slow is the GPU's batch size on a CPU) and answer it badly, since
nobody would run a 1000-crystal batch on CPU.

| | GPU | CPU |
|---|---|---|
| samples/arm | 2000 (a full production arm) | 1000 |
| batch size | 1000, fixed | **swept**: 10, 25, 50, 100, 200 |
| arms | 10 seeds | 5 batch sizes × 2 seeds |

What makes them comparable is the **per-sample work**: same SG/Z′, same two-stage
500-step schedule, same energy function. So the common currency is
`samples completed / wall seconds`.

**Compare throughput, never wall time.** The arms are deliberately different sizes.

## Reading the result

```bash
python summarize_bench.py --logs 'slurm-*.out'
```

It reports samples/s per arm, medians per device, **CPU throughput by batch size**
(which is the operating point the sweep exists to find), and the extrapolated cost
of 100,000 samples on each.

**A timed-out CPU arm is still a valid measurement.** `run_search` calls
`torch.save(opt_outs, out_path)` after every batch, so an arm killed at the 24h
wall leaves everything it finished, and `summarize_bench.py` counts what is on
disk rather than what was requested. An arm that did 600 of 1000 samples is a
perfectly good throughput number; only an arm with *zero* output tells us nothing.
That is what lets the CPU side ask for 1000 samples without risking a null result.

## Why the two scripts differ

Only in what the comparison requires:

| | GPU | CPU |
|---|---|---|
| `--gres` | `gpu:1` | none |
| singularity | `--nv` | **no `--nv`** |
| `--cpus-per-task` | 4 | 16 |
| `--mem` | 32G | 48G |
| `--time` | 4h | 24h |

**No `--nv` on the CPU job** is load-bearing: with it, a GPU is visible inside the
container and torch could use one despite `device: cpu`, silently invalidating the
comparison. `CUDA_VISIBLE_DEVICES=''` is set inside as well, and the job prints
`cuda_available=` before starting so the log proves it ran on CPU.

**Thread pinning** also matters: torch defaults to all *visible* cores, not the
ones SLURM allocated, so on a shared node an unpinned job oversubscribes and the
timing means nothing. `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are set from
`SLURM_CPUS_PER_TASK`, and the printed `torch threads=` confirms it.

**24h on the CPU side is a limit, not a target**, and hitting it is fine — see
the note above about partial arms. Only raise it if arms are producing *zero*
output.

## What to upload

Everything below goes to the cluster before either job will run.

| what | to | size |
|---|---|---|
| `opt_acridine_conformer.pt` | `/scratch/mk8347/data/crystal_datasets/acridine/` | 17 KB |
| `acr_112025_mh1_stagetwo.model` | `/scratch/mk8347/data/` | 57 MB |
| this repo (configs + `mxtaltools/`) | `/scratch/mk8347/projects/gfn_cond/MXtalTools` | — |

**Must exist, or the job exits before doing any work** (the scripts check):

```bash
mkdir -p /scratch/mk8347/data/crystal_datasets/acridine/opt_outs
```

**Not needed for these runs:** `esen_s.pt`. Every config carries a
`uma_predictor_path`, but `parse_opt_config` only loads it when
`optim_target == 'uma'`, and every arm here is `mace`. The field is inert.

**Not needed for the benchmark, but needed for the production battery's seeded
arms:** the 30 seed shards (`seeds_sg{14_zp1,14_zp2,9_zp2,5_zp2,19_zp3}_{0..5}.pt`,
110 MB total) to `/scratch/mk8347/data/crystal_datasets/acridine/`. The
`*_counts.json` sidecars are read by the config generator locally and do **not**
need uploading.

Assumed already present from previous runs — verify rather than trust:

```bash
ls /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
ls /scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
```

## One thing this cannot tell you

It measures the **search**. The rest of the workflow — collation, RDF distance
matrices, COMPACK — has a different profile: COMPACK is CPU-only regardless, and
the RDF work measured on CPU locally ran at ~48 structures/s, which is fast enough
that GPU buys little. So a bad CPU ratio here does not by itself rule out
offloading the whole pipeline; it rules out offloading the *optimization*.
