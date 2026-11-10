#!/usr/bin/env bash
# Submit the naive ReCom walk ensemble on the soal partition.
#
#   analysis/submit_recom_walks.sh OUTPUT_DIR [JOBS] [SEEDS_PER_BLOCK] [NUM_SEEDS] [ITERATIONS]
#
# The work is every centroid set x every block of SEEDS_PER_BLOCK seeds
# (default 21 x 20 = 420 blocks). The soal QOS allows a user 20 submitted and
# 12 running jobs, and an array task counts as a job, so the parallelism is
# inside the jobs: an array of JOBS (default 10) jobs, each taking every
# JOBS-th block (so slow and fast centroid sets mix) and running all of them
# at once on one CPU each. A first job syncs the environment; the array waits
# for it and never syncs, so the processes do not race on the same .venv.
#
# Each block writes OUTPUT_DIR/<centroids>_seeds<a>-<b>.{npz,json} and a log
# under OUTPUT_DIR/logs/; a block whose .json exists is skipped, so rerunning
# the same command finishes what failed. Run from the checkout the jobs
# should use, on sc. SKIP_SYNC=1 skips the environment sync job.
set -euo pipefail

OUTPUT_DIR=${1:?usage: $0 OUTPUT_DIR [JOBS] [SEEDS_PER_BLOCK] [NUM_SEEDS] [ITERATIONS]}
JOBS=${2:-10}
SEEDS_PER_BLOCK=${3:-5}
NUM_SEEDS=${4:-100}
ITERATIONS=${5:-1000}
# Each walk process briefly peaks near 3 GB while the data loads, and every
# block of a job starts together, so 3 GB a block is not enough: 40 of 420
# were OOM-killed at 3 GB.
MEM_PER_BLOCK_GB=${MEM_PER_BLOCK_GB:-6}
CONFIG=benchmark/configs/recom_walks/ideal_baseline_2627.yaml
# benchmark_edges' 20 centroid sets, plus one zone per school.
SETS="6-zone-3 6-zone-9 6-zone-1 6-zone-4 8-zone-22 8-zone-24 8-zone-25 8-zone-31
10-zone-3 10-zone-9 10-zone-1 10-zone-7 13-zone-6 13-zone-7 13-zone-2 13-zone-3
18-zone-1 18-zone-7 18-zone-5 18-zone-8 all"

read -r -a sets <<< "$(echo $SETS)"
chunks=$(( (NUM_SEEDS + SEEDS_PER_BLOCK - 1) / SEEDS_PER_BLOCK ))
blocks=$(( ${#sets[@]} * chunks ))
per_job=$(( (blocks + JOBS - 1) / JOBS ))
mkdir -p "$OUTPUT_DIR/logs"

# A block is "<set> <first>-<last>"; job j runs blocks j, j + JOBS, ...
: > "$OUTPUT_DIR/blocks.txt"
for set_name in "${sets[@]}"; do
  for (( c = 0; c < chunks; c++ )); do
    first=$(( c * SEEDS_PER_BLOCK + 1 ))
    last=$(( first + SEEDS_PER_BLOCK - 1 ))
    (( last > NUM_SEEDS )) && last=$NUM_SEEDS
    echo "$set_name $first-$last" >> "$OUTPUT_DIR/blocks.txt"
  done
done

dependency=()
if [[ -z "${SKIP_SYNC:-}" ]]; then
  sync_job=$(sbatch --parsable --account=soal --partition=soal \
    --cpus-per-task=1 --mem=4G --time=00:20:00 --job-name=recom-walks-sync \
    --output="$OUTPUT_DIR/logs/sync-%j.out" \
    --wrap "export PATH=\$HOME/.local/bin:\$PATH; cd $PWD && uv sync --frozen")
  echo "sync job $sync_job" >&2
  dependency=(--dependency=afterok:$sync_job)
fi

sbatch --parsable --account=soal --partition=soal "${dependency[@]}" \
  --array=0-$((JOBS - 1)) --cpus-per-task=$per_job --mem=$(( per_job * MEM_PER_BLOCK_GB ))G \
  --time=01:00:00 --job-name=recom-walks --output="$OUTPUT_DIR/logs/job-%A_%a.out" \
  --export=ALL,JOBS=$JOBS,ITERATIONS=$ITERATIONS,CONFIG=$CONFIG,OUTPUT_DIR=$OUTPUT_DIR,REPO=$PWD \
  <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
# One thread per walk process: the parallelism is the processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "job $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, at $(git rev-parse HEAD)"
awk -v j="$SLURM_ARRAY_TASK_ID" -v n="$JOBS" '(NR - 1) % n == j' "$OUTPUT_DIR/blocks.txt" |
  xargs -P "$SLURM_CPUS_PER_TASK" -L 1 bash -c '
    log="$OUTPUT_DIR/logs/$0_seeds$1.log"
    # Rerunnable: a block whose result exists is done.
    [[ -s "$OUTPUT_DIR/$0_seeds$1.json" ]] && { echo "skip $0 $1"; exit 0; }
    uv run --no-sync python analysis/recom_walk_ensemble.py "$CONFIG" --level Block_1 \
      --centroids "$0" --seeds "$1" --iterations "$ITERATIONS" -o "$OUTPUT_DIR" \
      > "$log" 2>&1 && echo "ok $0 $1" || { echo "FAILED $0 $1 (see $log)"; exit 1; }'
JOB
