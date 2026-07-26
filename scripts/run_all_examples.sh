#!/bin/bash
# ===========================================================================
# scripts/run_all_examples.sh
# ===========================================================================
# Master orchestration: run all 18 example compare.sh in order.
#
# Phase A: Elastic + Base (10) — elastic reference for regression
# Phase B: Viscoelastic (8)  — SLS elastic-limit regression vs elastic ref
#
# GPU cases auto-skip if no GPU (compare.sh checks nvidia-smi).
#
# Usage:
#   source scripts/env.sh && bash scripts/run_all_examples.sh
#   bash scripts/run_all_examples.sh --dry-run    # list cases only
#   bash scripts/run_all_examples.sh --phase A    # elastic only
#   bash scripts/run_all_examples.sh --phase B    # viscoelastic only
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/log/example_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.txt"

# ── Case list ──────────────────────────────────────────────
# Phase A: Elastic + Base (run first to generate greenfun reference)
PHASE_A_CASES=(
	"examples/halfspace" # base
	"examples/layer"     # base
	"examples/halfspace-elastic-mpi-cpp"
	"examples/halfspace-elastic-mpi-python"
	"examples/halfspace-elastic-gpu-cpp"
	"examples/halfspace-elastic-gpu-python"
	"examples/layer-elastic-mpi-cpp"
	"examples/layer-elastic-mpi-python"
	"examples/layer-elastic-gpu-cpp"
	"examples/layer-elastic-gpu-python"
	"examples/fullspace-cubic-elastic-mpi-cpp"
)

# Phase B: Viscoelastic (SLS elastic-limit regression)
PHASE_B_CASES=(
	"examples/halfspace-viscoelastic-mpi-cpp"
	"examples/halfspace-viscoelastic-mpi-python"
	"examples/halfspace-viscoelastic-gpu-cpp"
	"examples/halfspace-viscoelastic-gpu-python"
	"examples/layer-viscoelastic-mpi-cpp"
	"examples/layer-viscoelastic-mpi-python"
	"examples/layer-viscoelastic-gpu-cpp"
	"examples/layer-viscoelastic-gpu-python"
)

# ── Parse args ─────────────────────────────────────────────
DRY_RUN=false
PHASE_FILTER=""

for arg in "$@"; do
	case "$arg" in
	--dry-run) DRY_RUN=true ;;
	--phase)
		PHASE_FILTER="${2:-}"
		shift
		;;
	--phase=*) PHASE_FILTER="${arg#*=}" ;;
	-h | --help)
		echo "Usage: bash scripts/run_all_examples.sh [--dry-run] [--phase A|B]"
		echo ""
		echo "  --dry-run   List cases without executing"
		echo "  --phase A   Run elastic+base only (10 cases)"
		echo "  --phase B   Run viscoelastic only (8 cases)"
		exit 0
		;;
	esac
done

# ── Dry run ────────────────────────────────────────────────
if $DRY_RUN; then
	echo "=== Elastic + Base (Phase A) ==="
	for d in "${PHASE_A_CASES[@]}"; do echo "  $d"; done
	echo ""
	echo "=== Viscoelastic (Phase B) ==="
	for d in "${PHASE_B_CASES[@]}"; do echo "  $d"; done
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
if [ -z "$PHASE_FILTER" ] || [ "$PHASE_FILTER" = "A" ]; then
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo " PHASE A: Elastic + Base (10 cases)"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "" | tee -a "${SUMMARY_FILE}"
	echo "=== Phase A: Elastic + Base ===" | tee -a "${SUMMARY_FILE}"

	for case_dir in "${PHASE_A_CASES[@]}"; do
		run_case "$case_dir"
	done
fi

if [ -z "$PHASE_FILTER" ] || [ "$PHASE_FILTER" = "B" ]; then
	echo "" | tee -a "${SUMMARY_FILE}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo " PHASE B: Viscoelastic (8 cases) — SLS elastic-limit regression"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "" | tee -a "${SUMMARY_FILE}"
	echo "=== Phase B: Viscoelastic ===" | tee -a "${SUMMARY_FILE}"

	for case_dir in "${PHASE_B_CASES[@]}"; do
		run_case "$case_dir"
	done
fi

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
