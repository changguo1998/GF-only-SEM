#!/usr/bin/env bash
# ===========================================================================
# env_setup.sh — deprecated wrapper, use scripts/env.sh instead
# ===========================================================================
# This file is kept for backward compatibility. New scripts should use:
#   source scripts/env.sh
# ===========================================================================

ENV_SETUP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ENV_SETUP_SCRIPT_DIR}/scripts/env.sh"
