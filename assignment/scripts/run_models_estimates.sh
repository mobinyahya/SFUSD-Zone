#!/usr/bin/env bash
# run_models_estimates.sh
#
# Focused DA pipeline for SFUSD-Choice estimate models.
# Default settings (scripts/settings/models_cluster.env) reproduce the
# alternative / baseline / selected × k1/k3/k5 runs on years 2223 (in-sample) and 2324
# (out-of-sample), with list-length variants 0.8/0.7/0.6*real and 7, plus
# the status_quo_real reference runs.
#
# ALL paths and run-matrix values come from a sourced settings file —
# nothing machine-specific is hardcoded here. Environment variables set
# before invocation override the settings file (see the `: "${VAR:=...}"`
# pattern in scripts/settings/*.env).
#
# Pipeline steps:
#   1. Generate one run_custom_config.py YAML per entry  -> $CFG_DIR
#   2. Run the simulations in parallel                   -> $RUNS_ROOT
#   3. Generate the analyze_trends config                -> $ANALYSIS_CFG
#   4. Run scripts/analysis/analyze_trends.py            -> $OUTPUT_DIR
#   5. Add per-year sheets to the final Excel
#
# Usage:
#   bash scripts/run_models_estimates.sh [OPTIONS]
#
# Options:
#   --settings FILE   Settings file to source
#                     (default: scripts/settings/models_cluster.env)
#   --no-generate     Skip custom config generation
#   --no-simulate     Skip simulation step
#   --no-analyze      Skip analyze_trends step
#   --skip-existing   Skip simulations whose output folder already has results
#   --dry-run         Print commands without executing

set -euo pipefail

_cleanup_jobs() {
    local status=$?
    local job_pids
    trap - EXIT INT TERM
    job_pids="$(jobs -p)"
    if [[ -n "$job_pids" ]]; then
        kill $job_pids 2>/dev/null || true
    fi
    exit "$status"
}
trap _cleanup_jobs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# -- Defaults ----------------------------------------------------------------
DO_GENERATE=true
DO_SIMULATE=true
DO_ANALYZE=true
SKIP_EXISTING=false
DRY_RUN=false
SETTINGS_FILE="${SCRIPT_DIR}/settings/models_cluster.env"

# -- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --settings)      SETTINGS_FILE="$2";  shift 2 ;;
        --no-generate)   DO_GENERATE=false;   shift ;;
        --no-simulate)   DO_SIMULATE=false;   shift ;;
        --no-analyze)    DO_ANALYZE=false;    shift ;;
        --skip-existing) SKIP_EXISTING=true;  shift ;;
        --dry-run)       DRY_RUN=true;        shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "Settings file not found: $SETTINGS_FILE" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$SETTINGS_FILE"

: "${SIMULATION_WORKERS:=1}"
if [[ ! "$SIMULATION_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SIMULATION_WORKERS must be a positive integer: $SIMULATION_WORKERS" >&2
    exit 1
fi
if [[ ! "$ITER_START" =~ ^[0-9]+$ || ! "$ITER_END" =~ ^[0-9]+$ ]] \
    || (( ITER_END <= ITER_START )); then
    echo "ITER_START and ITER_END must define a nonempty integer range: ${ITER_START}..${ITER_END}" >&2
    exit 1
fi

# -- Helpers -----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Resolve a possibly-relative path against the project root.
_abs() {
    case "$1" in
        /*) echo "$1" ;;
        *)  echo "${PROJECT_ROOT}/${1#./}" ;;
    esac
}

ZONES_DIR_ABS="$(_abs "$ZONES_DIR")"
SCHOOL_DATA_DIR_ABS="$(_abs "$SCHOOL_DATA_DIR")"
SFUSD_MODELS_DIR_ABS="$(_abs "$SFUSD_MODELS_DIR")"

# ============================================================================
# ENTRIES TABLE — column order in the final Excel matches this order.
# Entry format (colon-separated):
#   RUN_LABEL : SFUSD_MODEL : TEST_YEAR : YEAR_INT : SUBCONFIG : LIST_LENGTH
# ============================================================================
ENTRIES=()

# status_quo_real reference runs (list-length irrelevant — utility off),
# one per TEST_SPECS year.
for test_spec in "${TEST_SPECS[@]}"; do
    IFS=':' read -r test_year year_int _sample_tag <<< "$test_spec"
    ENTRIES+=("status_quo_real_${test_year}::${test_year}:${year_int}:status_quo_real")
done

if [[ ${#MODEL_FAMILIES[@]} -gt 0 ]]; then
    for model_family in "${MODEL_FAMILIES[@]}"; do
        for k in "${K_VALUES[@]}"; do
            base="${model_family}_${TRAIN_YEAR}_k${k}_${MODEL_SUFFIX}"
            # For each test_year, emit one entry per list-length variant.
            for test_spec in "${TEST_SPECS[@]}"; do
                IFS=':' read -r test_year year_int sample_tag <<< "$test_spec"
                label_suffix=""
                [[ "$sample_tag" == "is" ]] && label_suffix="_${test_year}"
                for variant in "${LIST_LENGTH_VARIANTS[@]}"; do
                    ll_suffix="${variant%%:*}"
                    ll_expr="${variant#*:}"
                    run_label="${base}${label_suffix}_${ll_suffix}"
                    ENTRIES+=("${run_label}:${base}:${test_year}:${year_int}:status_quo:${ll_expr}")
                done
            done
        done
    done
fi
# ============================================================================

_parse_entry() {
    # Sets globals: RUN_LABEL SFUSD_MODEL TEST_YEAR YEAR_INT SUBCONFIG LIST_LENGTH
    IFS=':' read -r RUN_LABEL SFUSD_MODEL TEST_YEAR YEAR_INT SUBCONFIG LIST_LENGTH <<< "$1"
    if [[ -z "${LIST_LENGTH:-}" ]]; then
        LIST_LENGTH="0.8*round(real_length)"
    fi
}

_run_folder() { echo "$(_abs "$RUNS_ROOT")/${1}"; }

_is_done() {
    local run_label="$1"
    local subconfig="$2"
    local done_marker
    local iteration
    done_marker="$(_run_folder "$run_label")/${subconfig}"
    [[ -d "$done_marker" ]] || return 1

    for (( iteration=ITER_START; iteration<ITER_END; iteration++ )); do
        if [[ -z "$(find "$done_marker" -type f \
            -name "*_iteration${iteration}.csv" -print -quit 2>/dev/null)" ]]; then
            return 1
        fi
    done
    return 0
}

# -- Step 1: Generate custom configs -----------------------------------------
if $DO_GENERATE; then
    log "=== Step 1: Generating custom configs ==="
    mkdir -p "$CFG_DIR"

    for entry in "${ENTRIES[@]}"; do
        _parse_entry "$entry"
        cfg_file="${CFG_DIR}/${RUN_LABEL}.yaml"
        run_folder="$(_run_folder "$RUN_LABEL")"

        local_student="$(_abs "${STUDENT_DIR}/student_${TEST_YEAR}_filtered.csv")"
        local_program="$(_abs "${PROGRAM_DIR}/programs_without_specialprogs_${TEST_YEAR}.csv")"
        school_data="${SCHOOL_DATA_DIR_ABS}/schools_rehauled_${TEST_YEAR}.csv"

        if [[ -n "$SFUSD_MODEL" ]]; then
            estimate_path="${SFUSD_MODELS_DIR_ABS}/${SFUSD_MODEL}/estimates_${TEST_YEAR}.csv"
            enable_utility="true"
        else
            estimate_path=""
            enable_utility="false"
        fi

        if $DRY_RUN; then
            echo "  [DRY-RUN] Write: $cfg_file"
            continue
        fi

        log "  Writing: $cfg_file"
        cat > "$cfg_file" << YAML
grade: '${GRADE}'
iterations:
  end: ${ITER_END}
  start: ${ITER_START}
paths:
  assignment-folder: ${run_folder}/
  citywide-or-lp-zones:
    18zone_1_2-BG-0point3miles: ${ZONES_DIR_ABS}/18-zone_1_2-additiona-0point3miles.txt
  estimate-path: ${estimate_path}
  program-data: ${local_program}
  school-data: ${school_data}
  sfusd: ${SFUSD_DATA_DIR}
  student-data: ${local_student}
  student-save: ${run_folder}/precomputed/
  zone-files:
    10zone: ${ZONES_DIR_ABS}/10-zone-11_BG.csv
    13zone: ${ZONES_DIR_ABS}/13-zone-7_BG.csv
    18zone_1: ${ZONES_DIR_ABS}/18-zone-1_1_BG.csv
    18zone_2: ${ZONES_DIR_ABS}/18-zone-1_2_BG.csv
    59zone: ${ZONES_DIR_ABS}/59-zone-1_B.csv
    6zone-1: ${ZONES_DIR_ABS}/6-zone-1_BG.csv
    6zone-9_1: ${ZONES_DIR_ABS}/6-zone-9_1_BG.csv
    6zone-9_2: ${ZONES_DIR_ABS}/6-zone-9_2_BG.csv
    Con1: ${ZONES_DIR_ABS}/concept1zones.csv
r1-only: true
random-seed: ${RANDOM_SEED}
remove-special-lps: true
rounds-merged-options:
- 0
save-assignment: true
subconfigs:
- ${SUBCONFIG}
utility-model:
  designate-lp-for-all: false
  enable: ${enable_utility}
  list-length: "${LIST_LENGTH}"
  save-path: ${run_folder}/utility_matrix.csv
year: ${YEAR_INT}
YAML
    done
    log "  Generated ${#ENTRIES[@]} config candidate(s) in ${CFG_DIR}/"
fi

# -- Step 2: Simulate --------------------------------------------------------
if $DO_SIMULATE; then
    log "=== Step 2: Running simulations ==="

    mkdir -p "$LOG_DIR"

    # Validate every config before starting any work. Completed runs do not
    # require their source config when --skip-existing is active.
    if ! $DRY_RUN; then
        MISSING_CONFIGS=()
        for entry in "${ENTRIES[@]}"; do
            _parse_entry "$entry"
            if $SKIP_EXISTING && _is_done "$RUN_LABEL" "$SUBCONFIG"; then
                continue
            fi
            cfg_file="${CFG_DIR}/${RUN_LABEL}.yaml"
            if [[ ! -f "$cfg_file" ]]; then
                MISSING_CONFIGS+=("$cfg_file")
            fi
        done
        if [[ ${#MISSING_CONFIGS[@]} -gt 0 ]]; then
            log "ERROR: ${#MISSING_CONFIGS[@]} simulation config(s) are missing:"
            for cfg_file in "${MISSING_CONFIGS[@]}"; do
                echo "  - $cfg_file" >&2
            done
            exit 1
        fi
    fi

    PIDS=()
    PID_LABELS=()
    PID_SUBCONFIGS=()
    FAILED_LABELS=()

    _wait_for_simulation_batch() {
        local i pid label subconfig
        for (( i=0; i<${#PIDS[@]}; i++ )); do
            pid="${PIDS[$i]}"
            label="${PID_LABELS[$i]}"
            subconfig="${PID_SUBCONFIGS[$i]}"
            if ! wait "$pid"; then
                FAILED_LABELS+=("$label")
            elif ! _is_done "$label" "$subconfig"; then
                log "  ERROR: Incomplete assignment output: $label"
                FAILED_LABELS+=("$label")
            fi
        done
        PIDS=()
        PID_LABELS=()
        PID_SUBCONFIGS=()
    }

    log "  Simulation concurrency limit: $SIMULATION_WORKERS"
    for entry in "${ENTRIES[@]}"; do
        _parse_entry "$entry"
        cfg_file="${CFG_DIR}/${RUN_LABEL}.yaml"
        log_file="${LOG_DIR}/${RUN_LABEL}.log"

        if $SKIP_EXISTING && _is_done "$RUN_LABEL" "$SUBCONFIG"; then
            log "  SKIP (already done): $RUN_LABEL"
            continue
        fi

        log "  Starting: $RUN_LABEL"
        if $DRY_RUN; then
            echo "  [DRY-RUN] $PYTHON_CMD run_custom_config.py --config-path $cfg_file > $log_file 2>&1 &"
        else
            $PYTHON_CMD run_custom_config.py --config-path "$cfg_file" > "$log_file" 2>&1 &
            PIDS+=("$!")
            PID_LABELS+=("$RUN_LABEL")
            PID_SUBCONFIGS+=("$SUBCONFIG")
            if [[ ${#PIDS[@]} -ge "$SIMULATION_WORKERS" ]]; then
                _wait_for_simulation_batch
            fi
        fi
    done

    if ! $DRY_RUN; then
        log "  Waiting for remaining simulations to finish..."
        _wait_for_simulation_batch
        if [[ ${#FAILED_LABELS[@]} -gt 0 ]]; then
            log "ERROR: ${#FAILED_LABELS[@]} simulation(s) failed or produced incomplete output:"
            for lbl in "${FAILED_LABELS[@]}"; do echo "  - $lbl"; done
            log "Check logs in ${LOG_DIR}/"
            exit 1
        else
            log "  All simulations completed successfully."
        fi
    fi
fi

# -- Step 3: Generate analyze_trends config ----------------------------------
if $DO_ANALYZE; then
    log "=== Step 3: Generating analyze_trends config ==="
    mkdir -p "$(dirname "$ANALYSIS_CFG")"

    if ! $DRY_RUN; then
        cat > "$ANALYSIS_CFG" << HEADER
output_dir: ${OUTPUT_DIR}
HEADER

        if [[ -n "${ANALYSIS_NEW_CTIP_PATH:-}" ]]; then
            echo "new_ctip_path: $(_abs "$ANALYSIS_NEW_CTIP_PATH")" >> "$ANALYSIS_CFG"
        fi

        echo "" >> "$ANALYSIS_CFG"
        echo "runs:" >> "$ANALYSIS_CFG"

        RUNS_TO_ANALYZE=0
        INCOMPLETE_RUNS=()
        for entry in "${ENTRIES[@]}"; do
            _parse_entry "$entry"
            if ! _is_done "$RUN_LABEL" "$SUBCONFIG"; then
                log "  ERROR: Missing expected assignment iterations: $RUN_LABEL"
                INCOMPLETE_RUNS+=("$RUN_LABEL")
                continue
            fi
            run_folder="$(_run_folder "$RUN_LABEL")"
            local_student="$(_abs "${STUDENT_DIR}/student_${TEST_YEAR}_filtered.csv")"
            local_program="$(_abs "${PROGRAM_DIR}/programs_without_specialprogs_${TEST_YEAR}.csv")"
            school_data="${SCHOOL_DATA_DIR_ABS}/schools_rehauled_${TEST_YEAR}.csv"

            cat >> "$ANALYSIS_CFG" << YAML
  - label: "${RUN_LABEL}"
    folder: ${run_folder}/${SUBCONFIG}
    year: ${YEAR_INT}
    program_data: ${local_program}
    student_data: ${local_student}
    schools_data: ${school_data}

YAML
            (( RUNS_TO_ANALYZE++ )) || true
        done

        if [[ ${#INCOMPLETE_RUNS[@]} -gt 0 ]]; then
            log "ERROR: Refusing partial analysis; ${#INCOMPLETE_RUNS[@]} run(s) are incomplete."
            exit 1
        fi

        # Keeps the metric order identical across runs of this pipeline.
        cat >> "$ANALYSIS_CFG" << 'ROW_ORDER'
row_order:
  - "Distance Av (All Assigned)"
  - "Distance < 0.5 (All Assigned)"
  - "Distance > 3 (All Assigned)"
  - "#Schools above 10% district FRL"
  - "#Schools above 10% district FRL (Non-Designated)"
  - "#Schools above 15% district FRL"
  - "#Schools above 15% district FRL (Non-Designated)"
  - "AALPI in school with +10% FRL"
  - "AALPI in school with +15% FRL"
  - "#Students in schools above 10% district FRL"
  - "#Students in schools above 15% district FRL"
  - "Prop students in schools above +15% district FRL (All Assigned)"
  - "#GE above +10% district FRL"
  - "#GE above +15% district FRL"
  - "Dissimilarity (High FRL)"
  - "Black/White exposure to poverty"
  - "#Schools with +10% High Income (95292)"
  - "#Schools with +15% High Income (95292)"
  - "#Schools with -10% High Income (95292)"
  - "#Schools with -15% High Income (95292)"
  - "Dissimilarity (Income below 95292)"
  - "#GE programs that have 1-4 African American or Pacific Islander students"
  - "Unassigned"
  - "Designated"
  - "Designated or Unassigned"
  - "Prop Top 1 choice (All Assigned)"
  - "Prop Top 3 choice (All Assigned)"
  - "Top 1 in-zone choice (All Assigned)"
  - "Top 3 in-zone choice (All Assigned)"
  - "Prop Distance > 3 and Rank>=5 (All Assigned)"
  - "Variance of rank (All Assigned)"
  - "Variance of in-zone rank (All Assigned)"
  - "Variance of distance (All Assigned)"
  - "Top 3 in-zone non-desig choice All Assigned (non-CTIP)"
  - "Number of assigned students (non-CTIP)"
  - "Number of designated students (non-CTIP)"
  - "Number of unassigned students (non-CTIP)"
  - "Prop designated or unassigned students (non-CTIP)"
  - "Prop designated students (non-CTIP)"
  - "Prop designated students All Assigned (non-CTIP)"
  - "Distance Av All Assigned (Black)"
  - "Distance < 0.5 All Assigned (Black)"
  - "Distance > 3 All Assigned (Black)"
  - "Prop students in schools above +15% district FRL (Black)"
  - "Prop Top 1 non-desig choice All Assigned (Black)"
  - "Prop Top 3 non-desig choice All Assigned (Black)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (Black)"
  - "Distance Av All Assigned (Asian)"
  - "Distance < 0.5 All Assigned (Asian)"
  - "Distance > 3 All Assigned (Asian)"
  - "Prop students in schools above +15% district FRL (Asian)"
  - "Prop Top 1 non-desig choice All Assigned (Asian)"
  - "Prop Top 3 non-desig choice All Assigned (Asian)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (Asian)"
  - "Distance Av All Assigned (Hispanic)"
  - "Distance < 0.5 All Assigned (Hispanic)"
  - "Distance > 3 All Assigned (Hispanic)"
  - "Prop students in schools above +15% district FRL (Hispanic)"
  - "Prop Top 1 non-desig choice All Assigned (Hispanic)"
  - "Prop Top 3 non-desig choice All Assigned (Hispanic)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (Hispanic)"
  - "Distance Av All Assigned (White)"
  - "Distance < 0.5 All Assigned (White)"
  - "Distance > 3 All Assigned (White)"
  - "Prop students in schools above +15% district FRL (White)"
  - "Prop Top 1 non-desig choice All Assigned (White)"
  - "Prop Top 3 non-desig choice All Assigned (White)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (White)"
  - "Distance Av All Assigned (High FRL)"
  - "Distance < 0.5 All Assigned (High FRL)"
  - "Distance > 3 All Assigned (High FRL)"
  - "Prop students in schools above +15% district FRL (High FRL)"
  - "Prop Top 1 non-desig choice All Assigned (High FRL)"
  - "Prop Top 3 non-desig choice All Assigned (High FRL)"
  - "Prop Distance > 3 and (in-zone Rank>=5 or designated) (High FRL)"
  - "Distance Av All Assigned (Low FRL)"
  - "Distance < 0.5 All Assigned (Low FRL)"
  - "Distance > 3 All Assigned (Low FRL)"
  - "Prop students in schools above +15% district FRL (Low FRL)"
  - "Prop Top 1 non-desig choice All Assigned (Low FRL)"
  - "Prop Top 3 non-desig choice All Assigned (Low FRL)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (Low FRL)"
  - "Distance Av All Assigned (CTIP)"
  - "Distance < 0.5 All Assigned (CTIP)"
  - "Distance > 3 All Assigned (CTIP)"
  - "Prop students in schools above +15% district FRL (CTIP)"
  - "Prop Top 1 non-desig choice All Assigned (CTIP)"
  - "Prop Top 3 non-desig choice All Assigned (CTIP)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (CTIP)"
  - "Distance Av All Assigned (non-CTIP)"
  - "Distance < 0.5 All Assigned (non-CTIP)"
  - "Distance > 3 All Assigned (non-CTIP)"
  - "Prop students in schools above +15% district FRL (non-CTIP)"
  - "Prop Top 1 non-desig choice All Assigned (non-CTIP)"
  - "Prop Top 3 non-desig choice All Assigned (non-CTIP)"
  - "Prop Distance > 3 and (Rank>=5 or designated) (non-CTIP)"
ROW_ORDER

        log "  Wrote: $ANALYSIS_CFG"
    else
        echo "  [DRY-RUN] Write: $ANALYSIS_CFG"
    fi

    # -- Step 4: Run analyze_trends -------------------------------------------
    log "=== Step 4: Running analyze_trends ==="
    EXCEL_PATH="${OUTPUT_DIR}/metrics_comparison.xlsx"
    if $DRY_RUN; then
        echo "  [DRY-RUN] $PYTHON_CMD scripts/analysis/analyze_trends.py --config $ANALYSIS_CFG"
    else
        if [[ "${RUNS_TO_ANALYZE:-0}" -eq 0 ]]; then
            log "ERROR: No complete runs are available for analysis."
            exit 1
        fi
        rm -f "$EXCEL_PATH"
        if ! $PYTHON_CMD scripts/analysis/analyze_trends.py --config "$ANALYSIS_CFG"; then
            log "ERROR: analyze_trends failed."
            exit 1
        fi
        if [[ ! -f "$EXCEL_PATH" ]]; then
            log "ERROR: Analysis completed without producing $EXCEL_PATH"
            exit 1
        fi
    fi

    # -- Step 5: Add per-year sheets to the Excel -----------------------------
    if $DRY_RUN; then
        echo "  [DRY-RUN] Add per-year sheets to $EXCEL_PATH"
    elif [[ -f "$EXCEL_PATH" ]]; then
        log "=== Step 5: Adding per-year sheets to $EXCEL_PATH ==="

        # Write the label→test_year mapping from ENTRIES so the Python
        # snippet can split columns without re-parsing shell state.
        LABEL_YEAR_MAP="$(mktemp)"
        for entry in "${ENTRIES[@]}"; do
            _parse_entry "$entry"
            echo "${RUN_LABEL},${TEST_YEAR}" >> "$LABEL_YEAR_MAP"
        done

        $PYTHON_CMD - "$EXCEL_PATH" "$LABEL_YEAR_MAP" << 'PYSPLIT'
import sys
import pandas as pd

excel_path, map_path = sys.argv[1], sys.argv[2]

label_year = {}
with open(map_path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        lbl, yr = line.split(",", 1)
        label_year[lbl] = yr

xl = pd.ExcelFile(excel_path)
sheets = {name: xl.parse(name, index_col=0) for name in xl.sheet_names}

# Choose the "Mean ± Std" sheet as the base — most informative for humans.
base_name = "Mean ± Std" if "Mean ± Std" in sheets else xl.sheet_names[0]
base = sheets[base_name]

years = sorted({y for y in label_year.values()})
for year in years:
    cols = [c for c in base.columns if label_year.get(c) == year]
    if not cols:
        continue
    sheets[year] = base[cols]

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name)

print(f"  Added per-year sheets: {years}  (base={base_name!r})")
PYSPLIT
        rm -f "$LABEL_YEAR_MAP"
    else
        log "  SKIP Step 5 (Excel not found: $EXCEL_PATH)"
    fi
fi

log "=== Done ==="
echo ""
echo "Output locations:"
echo "  Custom configs  : ${CFG_DIR}/"
echo "  Simulation runs : ${RUNS_ROOT}/"
echo "  Simulation logs : ${LOG_DIR}/"
echo "  Analysis config : ${ANALYSIS_CFG}"
echo "  Excel output    : ${OUTPUT_DIR}/metrics_comparison.xlsx"
