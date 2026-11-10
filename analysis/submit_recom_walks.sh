#!/usr/bin/env bash
# Submit the naive ReCom walk ensemble as one Slurm array: every centroid set
# x every block of SEEDS_PER_TASK seeds is one 1-CPU task, so the whole
# ensemble runs at once. Run from the checkout the jobs should use, on sc.
#
#   analysis/submit_recom_walks.sh OUTPUT_DIR [SEEDS_PER_TASK] [NUM_SEEDS] [ITERATIONS]
#
# Each task writes OUTPUT_DIR/<centroids>_seeds<a>-<b>.{npz,json}; logs go to
# OUTPUT_DIR/logs/. The config is benchmark/configs/recom_walks/ideal_baseline_2627.yaml.
set -euo pipefail

OUTPUT_DIR=${1:?usage: $0 OUTPUT_DIR [SEEDS_PER_TASK] [NUM_SEEDS] [ITERATIONS]}
SEEDS_PER_TASK=${2:-5}
NUM_SEEDS=${3:-100}
ITERATIONS=${4:-1000}
CONFIG=benchmark/configs/recom_walks/ideal_baseline_2627.yaml
# benchmark_edges' 20 centroid sets, plus one zone per school.
SETS="6-zone-3 6-zone-9 6-zone-1 6-zone-4 8-zone-22 8-zone-24 8-zone-25 8-zone-31
10-zone-3 10-zone-9 10-zone-1 10-zone-7 13-zone-6 13-zone-7 13-zone-2 13-zone-3
18-zone-1 18-zone-7 18-zone-5 18-zone-8 all"

read -r -a sets <<< "$(echo $SETS)"
chunks=$(( (NUM_SEEDS + SEEDS_PER_TASK - 1) / SEEDS_PER_TASK ))
tasks=$(( ${#sets[@]} * chunks ))
mkdir -p "$OUTPUT_DIR/logs"

sbatch --parsable --account=soal --partition=soal \
  --array=0-$((tasks - 1)) --cpus-per-task=1 --mem=3G --time=00:45:00 \
  --job-name=recom-walks --output="$OUTPUT_DIR/logs/%A_%a.out" \
  --export=ALL,SETS="$SETS",CHUNKS=$chunks,SEEDS_PER_TASK=$SEEDS_PER_TASK,NUM_SEEDS=$NUM_SEEDS,ITERATIONS=$ITERATIONS,CONFIG=$CONFIG,OUTPUT_DIR=$OUTPUT_DIR,REPO=$PWD \
  <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
read -r -a sets <<< "$(echo $SETS)"
id=$SLURM_ARRAY_TASK_ID
set_name=${sets[$(( id / CHUNKS ))]}
first=$(( (id % CHUNKS) * SEEDS_PER_TASK + 1 ))
last=$(( first + SEEDS_PER_TASK - 1 ))
(( last > NUM_SEEDS )) && last=$NUM_SEEDS
cd "$REPO"
echo "task $id: $set_name seeds $first-$last on $(hostname) at $(git rev-parse HEAD)"
uv run python analysis/recom_walk_ensemble.py "$CONFIG" --level Block_1 \
  --centroids "$set_name" --seeds "$first-$last" --iterations "$ITERATIONS" \
  -o "$OUTPUT_DIR"
JOB
