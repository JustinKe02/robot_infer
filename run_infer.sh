#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[tk_infer] conda executable was not found" >&2
  exit 2
fi

CONDA_ROOT="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  echo "[tk_infer] conda environment python was not found: ${CONDA_PYTHON}" >&2
  exit 2
fi

: "${POLICY_PATH:?Set POLICY_PATH to a local checkpoint/.../pretrained_model directory}"
: "${DATASET_ROOT:?Set DATASET_ROOT to a local LeRobot dataset root}"
: "${DATASET_REPO_ID:?Set DATASET_REPO_ID to the dataset repo id stored in metadata}"
SAMPLE_INDICES="${SAMPLE_INDICES:-first,middle,last}"
DEVICE="${DEVICE:-auto}"

for argument in "$@"; do
  case "${argument}" in
    --policy-path|--policy-path=*|--dataset-root|--dataset-root=*|--dataset-repo-id|--dataset-repo-id=*|\
    --sample-indices|--sample-indices=*|--device|--device=*|--tokenizer-path|--tokenizer-path=*|\
    --output-json|--output-json=*)
      echo "[tk_infer] configure fixed arguments through environment variables, not extra CLI args" >&2
      exit 2
      ;;
  esac
done

OUTPUT_JSON="${OUTPUT_JSON:-${TASK_ROOT}/outputs/inference_$(date +%Y%m%d_%H%M%S).json}"
OUTPUT_JSON_RESOLVED="$(readlink -m -- "${OUTPUT_JSON}")"
case "${OUTPUT_JSON_RESOLVED}" in
  "${TASK_ROOT}"|"${TASK_ROOT}"/*)
    ;;
  *)
    echo "[tk_infer] OUTPUT_JSON must stay inside ${TASK_ROOT}" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "${OUTPUT_JSON_RESOLVED}")"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "[tk_infer] environment=${CONDA_ENV} python=${CONDA_PYTHON}"
echo "[tk_infer] policy=${POLICY_PATH}"
echo "[tk_infer] dataset=${DATASET_ROOT} repo_id=${DATASET_REPO_ID}"
echo "[tk_infer] samples=${SAMPLE_INDICES} device=${DEVICE}"
echo "[tk_infer] hardware=disabled output=${OUTPUT_JSON_RESOLVED}"

COMMAND=(
  "${CONDA_PYTHON}"
  "${TASK_ROOT}/offline_infer.py"
  "--policy-path=${POLICY_PATH}"
  "--dataset-root=${DATASET_ROOT}"
  "--dataset-repo-id=${DATASET_REPO_ID}"
  "--sample-indices=${SAMPLE_INDICES}"
  "--device=${DEVICE}"
  "--output-json=${OUTPUT_JSON_RESOLVED}"
)
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  echo "[tk_infer] tokenizer=${TOKENIZER_PATH}"
  COMMAND+=("--tokenizer-path=${TOKENIZER_PATH}")
fi

exec "${COMMAND[@]}" "$@"
