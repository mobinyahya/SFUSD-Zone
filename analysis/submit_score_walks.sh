#!/usr/bin/env bash
# Score every state of the naive ReCom walk ensemble on the soal partition.
#
#   analysis/submit_score_walks.sh WALK_DIR [OUT_DIR]
#
# The work is one task per walk (centroid set x seed, 2,100 for the full
# ensemble), each running analysis/score_recom_walks.py --seed on its file.
# The soal QOS allows a user 12 running jobs, so the parallelism is inside the
# jobs: BIG_JOBS jobs of BIG_CPUS CPUs (for the 256-core nodes) and SMALL_JOBS
# of SMALL_CPUS (for the 64-core ones), every CPU running one worker. Workers
# share one queue: a worker claims a task by creating its directory under
# OUT_DIR/claims (mkdir is atomic), so jobs of any size stay busy until the
# queue is empty, and a task whose output exists is skipped, so rerunning
# the same command finishes what failed.
#
# A one-CPU smoke job scores three states of one walk first; the scoring jobs
# start only if it succeeds. Run from the checkout the jobs should use, on sc.
set -euo pipefail

WALK_DIR=${1:?usage: $0 WALK_DIR [OUT_DIR]}
OUT_DIR=${2:-$WALK_DIR/scoring/states}
BIG_JOBS=${BIG_JOBS:-7}
BIG_CPUS=${BIG_CPUS:-120}
SMALL_JOBS=${SMALL_JOBS:-5}
SMALL_CPUS=${SMALL_CPUS:-56}
MEM_PER_CPU=${MEM_PER_CPU:-5G}
CONFIG=benchmark/configs/recom_walks/score_ideal_baseline_2627.yaml
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/claims"
# A rerun: a claim without its output is a task that failed or was killed.
for claim in "$OUT_DIR"/claims/*/; do
  [[ -d "$claim" ]] || continue
  name=$(basename "$claim")
  [[ -s "$OUT_DIR/$name.npz" ]] || rmdir "$claim"
done

# One task per walk: "<walk file> <seed>".
python3 - "$WALK_DIR" > "$OUT_DIR/tasks.txt" <<'PY'
import glob, os, re, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_seeds*-*.npz"))):
    first, last = map(int, re.search(r"_seeds(\d+)-(\d+)\.npz$", path).groups())
    for seed in range(first, last + 1):
        print(path, seed)
PY
echo "$(wc -l < "$OUT_DIR/tasks.txt") walks to score" >&2

common=(--account=soal --partition=soal --export=ALL,OUT_DIR=$OUT_DIR,CONFIG=$CONFIG,REPO=$PWD)
first_walk=$(head -1 "$OUT_DIR/tasks.txt")
smoke=$(sbatch --parsable "${common[@]}" --cpus-per-task=1 --mem=8G --time=00:30:00 \
  --job-name=score-walks-smoke --output="$OUT_DIR/logs/smoke-%j.out" \
  --wrap "export PATH=\$HOME/.local/bin:\$PATH OMP_NUM_THREADS=1; cd $PWD && \
    uv run --no-sync python -m analysis.score_recom_walks $CONFIG ${first_walk% *} \
      --seed ${first_walk#* } --every 500 -o $OUT_DIR/smoke")
echo "smoke job $smoke" >&2

job_script=$(cat <<'JOB'
#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)
cd "$REPO"
echo "job ${SLURM_JOB_ID} on $(hostname), $SLURM_CPUS_PER_TASK workers, at $(git rev-parse --short HEAD)"
worker() {
  while read -r walk seed; do
    name=$(basename "$walk" .npz); name="${name%%_seeds*}_seed$seed"
    [[ -s "$OUT_DIR/$name.npz" ]] && continue
    mkdir "$OUT_DIR/claims/$name" 2>/dev/null || continue
    if uv run --no-sync python -m analysis.score_recom_walks "$CONFIG" "$walk" \
        --seed "$seed" -o "$OUT_DIR" < /dev/null > "$OUT_DIR/logs/$name.log" 2>&1; then
      echo "ok $name"
    else
      echo "FAILED $name (see $OUT_DIR/logs/$name.log)"
    fi
  done < "$OUT_DIR/tasks.txt"
}
for (( w = 0; w < SLURM_CPUS_PER_TASK; w++ )); do worker & done
wait
echo "job ${SLURM_JOB_ID} done"
JOB
)

for spec in "big $BIG_JOBS $BIG_CPUS" "small $SMALL_JOBS $SMALL_CPUS"; do
  read -r kind jobs cpus <<< "$spec"
  (( jobs > 0 )) || continue
  sbatch --parsable "${common[@]}" --dependency=afterok:$smoke \
    --array=0-$((jobs - 1)) --cpus-per-task=$cpus --mem-per-cpu=$MEM_PER_CPU \
    --time=04:00:00 --job-name=score-walks-$kind \
    --output="$OUT_DIR/logs/job-$kind-%A_%a.out" <<< "$job_script"
done
