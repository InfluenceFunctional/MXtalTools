"""
CPU-vs-GPU throughput benchmark for the acridine search, sg14 Z'=2.

The question is practical -- "can we offload this to CPU?" -- so each device is
run at ITS OWN sensible operating point rather than on identical configs. Matching
them exactly would answer a different question (how much slower is the GPU's batch
size on a CPU) and would answer it badly, since a 1000-crystal batch is not how
anyone would run this on CPU.

  GPU   10 arms at a full production arm: 2000 samples, batch 1000, seeds 0-9.
        Fixed configuration, varying only the seed, so the spread is the
        node-to-node and starting-structure variance at the point we actually
        run.

  CPU   10 arms of at most 1000 samples, SWEEPING batch size over
        {10, 25, 50, 100, 200} x 2 seeds. The CPU's best batch size is unknown
        and is not the GPU's -- so the sweep measures throughput and finds the
        operating point in the same run.

WHAT MAKES THIS COMPARABLE. Not the arm size -- the per-sample work. Every arm
runs the same SG/Z', the same two-stage schedule (500 steps) and the same energy
function, so `samples completed / wall seconds` is the common currency. Compare
throughput, never wall time.

PARTIAL ARMS STILL COUNT. `run_search` calls torch.save(opt_outs, out_path) after
every batch, so an arm killed at the wall limit leaves everything it finished.
summarize_bench.py counts what is on disk rather than what was requested, so a
CPU arm that times out still yields a throughput number. That is why the CPU side
can afford 1000 samples.

    python make_bench_configs.py
    sbatch submit_bench_gpu.sbatch
    sbatch submit_bench_cpu.sbatch
    python summarize_bench.py            # after both finish
"""
from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = '../acr_rerun_aug21/acridine.yaml'
SG, ZP = 14, 2

#: GPU: a full production arm, repeated over seeds
GPU_ARMS = 10
GPU_SAMPLES = 2000
GPU_BATCH = 1000

#: CPU: capped samples, batch size swept. 5 sizes x 2 seeds = 10 arms.
CPU_SAMPLES = 1000
CPU_BATCHES = [10, 25, 50, 100, 200]
CPU_SEEDS = [0, 1]


def main():
    base = yaml.safe_load(Path(BASE_PATH).read_text())
    for old in Path('.').glob('bench_*.yaml'):
        old.unlink()

    def write(device, k, samples, batch, seed):
        cfg = deepcopy(base)
        cfg['device'] = 'cuda' if device == 'gpu' else 'cpu'
        cfg['sgs_to_search'] = [SG]
        cfg['zp_to_search'] = [ZP]
        cfg['num_samples'] = samples
        cfg['batch_size'] = batch
        cfg['opt_seed'] = seed
        cfg['init_target_cp'] = 'std'      # as acr_production does for Z'>1
        cfg['save_trajs'] = False          # not measuring I/O
        #: batch size is in the run_name so the sweep is readable from the
        #: filenames alone, and so summarize_bench.py can recover it without
        #: reopening every config
        cfg['run_name'] = f'bench{device}_acridine_sg{SG}_zp{ZP}_b{batch}_{k}'
        Path(f'bench_{device}_{k}.yaml').write_text(
            yaml.dump(cfg, default_flow_style=False))
        return batch, samples, seed

    print(f"GPU: {GPU_ARMS} arms, {GPU_SAMPLES} samples, batch {GPU_BATCH}")
    for k in range(GPU_ARMS):
        write('gpu', k, GPU_SAMPLES, GPU_BATCH, k)

    print(f"CPU: {len(CPU_BATCHES) * len(CPU_SEEDS)} arms, {CPU_SAMPLES} samples, "
          f"batch swept over {CPU_BATCHES}")
    k = 0
    for batch in CPU_BATCHES:
        for seed in CPU_SEEDS:
            write('cpu', k, CPU_SAMPLES, batch, seed)
            print(f"   bench_cpu_{k}.yaml  batch={batch:4d} seed={seed}")
            k += 1

    steps = sum(s['max_num_steps'] for s in base['opt'])
    print(f"\nboth devices: sg{SG} Z'={ZP}, {steps} steps over {len(base['opt'])} "
          f"stages, same energy function")
    print("compare THROUGHPUT (samples/second), not wall time -- the arms are "
          "deliberately different sizes")


if __name__ == '__main__':
    main()
