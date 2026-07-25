#!/usr/bin/env bash
set -euo pipefail

KERNEL_NAME="tinker-utm-voms"
DISPLAY_NAME="Tinker UTM+VOMS (Qwen)"

python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"

echo "Installed Jupyter kernel: ${DISPLAY_NAME} (${KERNEL_NAME})"
