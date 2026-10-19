#!/usr/bin/env bash
# scripts/run_provider_evals.sh
#
# Run evals.run_eval for anthropic and ollama providers across dev, test, and heldout splits.
# Results are written to evals/results/<provider>_<split>.{json,md}.
#
# Usage:
#   bash scripts/run_provider_evals.sh
#
# Requirements:
#   - Set ANTHROPIC_API_KEY environment variable to run Anthropic evaluations.
#   - Run a local Ollama daemon (http://localhost:11434) for Ollama evaluations.
#   - Set CALL_ANTHROPIC_MODEL to override the Anthropic model (default: claude-sonnet-5-5).
#   - Set CALL_OLLAMA_MODEL to override the Ollama model (default: llama3).
#
# ⚠️  SIMULATED DATA WARNING: All evaluation transcripts are SYNTHETIC/SIMULATED.
#     No real customer recordings or personal data are involved.

set -euo pipefail

RESULTS_DIR="evals/results"
SPLITS=("dev" "test" "heldout")

mkdir -p "${RESULTS_DIR}"

echo "======================================================="
echo " Call Center Insights — Provider Evaluation Runner"
echo " All data is SIMULATED / SYNTHETIC"
echo "======================================================="
echo ""

# ---------------------------------------------------------------------------
# Anthropic evaluation
# ---------------------------------------------------------------------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "SKIPPING anthropic evals: ANTHROPIC_API_KEY is not set."
    echo "  To run: export ANTHROPIC_API_KEY=<your key> && bash scripts/run_provider_evals.sh"
    echo ""
else
    MODEL="${CALL_ANTHROPIC_MODEL:-claude-sonnet-5-5}"
    echo "Running Anthropic evaluations (model: ${MODEL})..."
    for SPLIT in "${SPLITS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/anthropic_${SPLIT}.json"
        OUT_MD="${RESULTS_DIR}/anthropic_${SPLIT}.md"
        echo "  [anthropic] split=${SPLIT} → ${OUT_JSON}"
        python -m evals.run_eval \
            --provider anthropic \
            --split "${SPLIT}" \
            --no-audio \
            --out-json "${OUT_JSON}" \
            --out-md  "${OUT_MD}" \
            || echo "  WARNING: anthropic ${SPLIT} eval failed (see stderr above)"
    done
    echo "Anthropic evals complete."
    echo ""
fi

# ---------------------------------------------------------------------------
# Ollama evaluation
# ---------------------------------------------------------------------------
OLLAMA_HEALTHY=false
if curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    OLLAMA_HEALTHY=true
fi

if [ "${OLLAMA_HEALTHY}" = "false" ]; then
    echo "SKIPPING ollama evals: No Ollama daemon found at http://localhost:11434."
    echo "  To run: start Ollama (https://ollama.com) then re-run this script."
    echo ""
else
    MODEL="${CALL_OLLAMA_MODEL:-llama3}"
    echo "Running Ollama evaluations (model: ${MODEL})..."
    for SPLIT in "${SPLITS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/ollama_${SPLIT}.json"
        OUT_MD="${RESULTS_DIR}/ollama_${SPLIT}.md"
        echo "  [ollama] split=${SPLIT} → ${OUT_JSON}"
        python -m evals.run_eval \
            --provider ollama \
            --split "${SPLIT}" \
            --no-audio \
            --out-json "${OUT_JSON}" \
            --out-md  "${OUT_MD}" \
            || echo "  WARNING: ollama ${SPLIT} eval failed (see stderr above)"
    done
    echo "Ollama evals complete."
    echo ""
fi

echo "======================================================="
echo " All evaluations finished."
echo " Results written to ${RESULTS_DIR}/"
echo "======================================================="
