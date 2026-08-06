#!/bin/bash
# ===========================================================================
# scripts/run_all_examples.sh
# ===========================================================================
# Master orchestration: run the canonical examples end-to-end.
#
#   examples/halfspace/        homogeneous half-space vs analytic Lamb reference
#   examples/layer/            two-layer half-space vs PyFK reference
#   examples/fullspace-cubic/  full-space — solver / backend (GPU vs CPU) comparison
#
# Each compare.sh runs:
#   mesh → preprocess (GLL+material+PML+SLS attenuation) → forward →
#   postprocess (Green's function extraction) → verify vs reference
#
# The solver is chosen by the user inside examples/*/forward.sh (commented-out,
# switchable) — no test-case naming for GPU/CPU or solver anymore.
#
# GPU cases (fullspace-cubic) auto-skip if no GPU (compare.sh checks nvidia-smi).
#
# Usage:
#   source scripts/env.sh && bash scripts/run_all_examples.sh
#   bash scripts/run_all_examples.sh --dry-run               # list cases only
#   bash scripts/run_all_examples.sh --case halfspace        # run one case
#   bash scripts/run_all_examples.sh --case layer,fullspace  # run several
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/log/example_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.txt"

# ── Case list ──────────────────────────────────────────────
ALL_CASES=(
	"examples/halfspace"
	"examples/layer"
	"examples/fullspace-cubic"
)

# ── Parse args ─────────────────────────────────────────────
DRY_RUN=false
CASE_FILTER=""

while [ $# -gt 0 ]; do
	case "$1" in
	--dry-run)
		DRY_RUN=true
		shift
		;;
	--case)
		CASE_FILTER="${2:-}"
		shift 2
		;;
	--case=*)
		CASE_FILTER="${1#*=}"
		shift
		;;
	-h | --help)
		echo "Usage: bash scripts/run_all_examples.sh [--dry-run] [--case NAME[,NAME...]]"
		echo ""
		echo "  --dry-run        List cases without executing"
		echo "  --case NAME      Run only the named case(s): halfspace, layer, fullspace"
		exit 0
		;;
	*)
		echo "ERROR: unknown argument '$1'"
		exit 1
		;;
	esac
done

# Resolve --case filter to a list of case directories.
CASES=()
if [ -n "$CASE_FILTER" ]; then
	for name in $(echo "$CASE_FILTER" | tr ',' ' '); do
		case "$name" in
		halfspace) CASES+=("examples/halfspace") ;;
		layer) CASES+=("examples/layer") ;;
		fullspace | fullspace-cubic) CASES+=("examples/fullspace-cubic") ;;
		*)
			echo "ERROR: unknown case '$name' (valid: halfspace, layer, fullspace)"
			exit 1
			;;
		esac
	done
else
	CASES=("${ALL_CASES[@]}")
fi

# ── Dry run ────────────────────────────────────────────────
if $DRY_RUN; then
	echo "=== Examples ==="
	for d in "${CASES[@]}"; do echo "  $d"; done
	exit 0
fi

# ── Setup ──────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# Source project environment (no pipe — needs PATH for MPI)
source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true
export PATH

echo "Run started: $(date)" | tee "${SUMMARY_FILE}"
echo "Project: ${PROJECT_ROOT}" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

# ── Colour helpers ─────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

declare -A RESULTS
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
ORDERED_NAMES=()

run_case() {
	local case_dir="$1"
	local case_name
	case_name="$(basename "$case_dir")"
	ORDERED_NAMES+=("$case_name")
	TOTAL=$((TOTAL + 1))

	local log_file="${LOG_DIR}/${case_name}_${TIMESTAMP}.log"
	local compare_script="${PROJECT_ROOT}/${case_dir}/compare.sh"

	echo -n "[${TOTAL}] ${case_name} ... " | tee -a "${SUMMARY_FILE}"

	if [ ! -f "${compare_script}" ]; then
		echo -e "${RED}MISSING${NC} (no compare.sh)" | tee -a "${SUMMARY_FILE}"
		RESULTS["$case_name"]="MISSING"
		FAILED=$((FAILED + 1))
		return
	fi

	cd "${PROJECT_ROOT}/${case_dir}"

	# Use wrapper script that properly sources spack env before running
	if bash "${PROJECT_ROOT}/scripts/_run_case.sh" "${PROJECT_ROOT}/${case_dir}" >"${log_file}" 2>&1; then
		# Check if it was a GPU skip
		if grep -q "SKIP: no GPU available" "${log_file}" 2>/dev/null; then
			echo -e "${YELLOW}SKIP (no GPU)${NC}" | tee -a "${SUMMARY_FILE}"
			RESULTS["$case_name"]="SKIP"
			SKIPPED=$((SKIPPED + 1))
		else
			echo -e "${GREEN}PASSED${NC}" | tee -a "${SUMMARY_FILE}"
			RESULTS["$case_name"]="PASSED"
			PASSED=$((PASSED + 1))
		fi
	else
		local exit_code=$?
		# Check for GPU skip
		if grep -q "SKIP: no GPU available" "${log_file}" 2>/dev/null; then
			echo -e "${YELLOW}SKIP (no GPU)${NC}" | tee -a "${SUMMARY_FILE}"
			RESULTS["$case_name"]="SKIP"
			SKIPPED=$((SKIPPED + 1))
		else
			echo -e "${RED}FAILED (exit=${exit_code})${NC}" | tee -a "${SUMMARY_FILE}"
			echo "  Last 5 lines of log:" | tee -a "${SUMMARY_FILE}"
			tail -5 "${log_file}" | sed 's/^/    /' | tee -a "${SUMMARY_FILE}"
			RESULTS["$case_name"]="FAILED"
			FAILED=$((FAILED + 1))
		fi
	fi
}

# ── Run ─────────────────────────────────────────────────────
echo "" | tee -a "${SUMMARY_FILE}"
echo "=== Examples ($((${#CASES[@]})) cases) ===" | tee -a "${SUMMARY_FILE}"

for case_dir in "${CASES[@]}"; do
	run_case "$case_dir"
done

# ── Summary ─────────────────────────────────────────────────
echo "" | tee -a "${SUMMARY_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "" | tee -a "${SUMMARY_FILE}"

for name in "${ORDERED_NAMES[@]}"; do
	result="${RESULTS[$name]}"
	case "$result" in
	PASSED) echo -e "  ${GREEN}[PASS]${NC} $name" | tee -a "${SUMMARY_FILE}" ;;
	FAILED) echo -e "  ${RED}[FAIL]${NC} $name" | tee -a "${SUMMARY_FILE}" ;;
	SKIP) echo -e "  ${YELLOW}[SKIP]${NC} $name" | tee -a "${SUMMARY_FILE}" ;;
	MISSING) echo -e "  ${RED}[MISS]${NC} $name" | tee -a "${SUMMARY_FILE}" ;;
	*) echo "  [??] $name" | tee -a "${SUMMARY_FILE}" ;;
	esac
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "Total: ${TOTAL}  |  Passed: ${PASSED}  |  Failed: ${FAILED}  |  Skipped: ${SKIPPED}" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}/" | tee -a "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
echo "Run finished: $(date)" | tee -a "${SUMMARY_FILE}"

# ── Exit code ───────────────────────────────────────────────
if [ "$FAILED" -gt 0 ]; then
	exit 1
else
	exit 0
fi
