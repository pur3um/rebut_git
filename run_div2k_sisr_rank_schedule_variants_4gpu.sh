#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# DIV2K SISR rank-schedule grid on four 24 GB GPUs
#
# - Uses physical GPUs 0,1,2,3 by default.
# - Runs one experiment at a time on each GPU.
# - Covers both DIV2K_train_HR and DIV2K_valid_HR.
# - Cycles ReLU MLP, SIREN, real-valued WIRE, and FINER by default.
# - Uses model-specific Adam/auxiliary learning rates:
#     ReLU=1e-3, SIREN=1e-3, real WIRE=3e-2, FINER=3e-4.
# - Cycles Muon learning rates 2e-3 and 3e-3 by default.
# - Cycles all seven optimizers in rank_schedule_variants with rank_wsd.
# - A repeated command resumes from .done markers.
#
# Quick checks:
#   DRY_RUN=1 MAX_IMAGES_PER_SPLIT=1 EPOCHS="2000" \
#       bash run_div2k_sisr_rank_schedule_variants_4gpu.sh
#
# Full default grid:
#   bash run_div2k_sisr_rank_schedule_variants_4gpu.sh
#
# Several starting ranks:
#   RANK_STARTS="32 64 128" \
#       bash run_div2k_sisr_rank_schedule_variants_4gpu.sh
#
# Selected models only:
#   MODELS="siren_mlp real_wire finer_mlp" \
#       bash run_div2k_sisr_rank_schedule_variants_4gpu.sh
#
# A single Muon learning rate only:
#   MUON_LR="3e-3" \
#       bash run_div2k_sisr_rank_schedule_variants_4gpu.sh
# ============================================================================

# ---------------------------------------------------------------------------
# Repository, data, and output paths
# ---------------------------------------------------------------------------
BASE_PATH=${BASE_PATH:-/workspace/rebut_git}
FIT_SISR=${FIT_SISR:-"${BASE_PATH}/fit_sisr_rebut.py"}
TRAIN_ROOT=${TRAIN_ROOT:-"${BASE_PATH}/data/div2k/DIV2K_train_HR"}
VALID_ROOT=${VALID_ROOT:-"${BASE_PATH}/data/div2k/DIV2K_valid_HR"}
OUT_ROOT=${OUT_ROOT:-"${BASE_PATH}/results/div2k_sisr_rank_schedule_variants"}
PYTHON_BIN=${PYTHON_BIN:-python}

RESULTS_CSV=${RESULTS_CSV:-"${OUT_ROOT}/all_image_results.csv"}
SUMMARY_CSV=${SUMMARY_CSV:-"${OUT_ROOT}/summary_by_condition.csv"}
STATUS_FILE=${STATUS_FILE:-"${OUT_ROOT}/run_status.tsv"}
MANIFEST_FILE=${MANIFEST_FILE:-"${OUT_ROOT}/job_manifest.tsv"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-div2k_sisr_rank_schedule_variants}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
DATASETS=${DATASETS:-"train valid"}
OPTIMIZERS=${OPTIMIZERS:-"auto_cos_inc auto_exp_inc auto_log_inc auto_linear_inc auto_logistic_inc auto_cos_dec auto_fixed"}
SCHEDULER=${SCHEDULER:-rank_wsd}
EPOCHS=${EPOCHS:-"2000 3000"}
SCALE_FACTORS=${SCALE_FACTORS:-"4"}
SEEDS=${SEEDS:-"42"}

# Increasing variants: RANK_STARTS -> RANK_ENDS
# auto_cos_dec:        RANK_ENDS -> RANK_STARTS
# auto_fixed:          FIXED_RANKS
RANK_STARTS=${RANK_STARTS:-"64"}
RANK_ENDS=${RANK_ENDS:-"256"}
FIXED_RANKS=${FIXED_RANKS:-"200"}
RANK_FLOOR=${RANK_FLOOR:-}

RANK_WARMUP_STEPS=${RANK_WARMUP_STEPS:-auto}
RANK_WARMUP_RATIO=${RANK_WARMUP_RATIO:-0.8}
RANK_WSD_WARMUP_STEPS=${RANK_WSD_WARMUP_STEPS:-0}
RANK_WSD_DECAY_START_STEP=${RANK_WSD_DECAY_START_STEP:-auto}
RANK_WSD_DECAY_START_RATIO=${RANK_WSD_DECAY_START_RATIO:-${RANK_WARMUP_RATIO}}
RANK_WSD_MIN_LR_RATIO=${RANK_WSD_MIN_LR_RATIO:-0.1}

# Model-specific Adam/auxiliary learning rates. LR remains the fallback and
# the backward-compatible override for ReLU or any newly added model.
LR=${LR:-1e-3}
RELU_LR=${RELU_LR:-${LR}}
SIREN_LR=${SIREN_LR:-1e-3}
WIRE_LR=${WIRE_LR:-0.03}
FINER_LR=${FINER_LR:-3e-4}

# MUON_LRS defines a grid. For backward compatibility, setting MUON_LR without
# MUON_LRS runs only that single value.
MUON_LRS=${MUON_LRS:-}
if [[ -z "${MUON_LRS}" ]]; then
    MUON_LRS=${MUON_LR:-"2e-3 3e-3"}
fi
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_NS_STEPS=${MUON_NS_STEPS:-5}
EXP_RATE=${EXP_RATE:-5.0}
LOG_RATE=${LOG_RATE:-9.0}
LOGISTIC_STEEPNESS=${LOGISTIC_STEEPNESS:-12.0}
LOG_N_EPOCHS=${LOG_N_EPOCHS:-500}

# MODELS controls the model grid. MODEL remains a backward-compatible
# single-model override when MODELS is not explicitly provided.
MODELS=${MODELS:-}
if [[ -z "${MODELS}" ]]; then
    MODELS=${MODEL:-"relu_mlp siren_mlp real_wire finer_mlp"}
fi
NUM_LAYERS=${NUM_LAYERS:-5}
HIDDEN_DIM=${HIDDEN_DIM:-300}
SR_TRAIN_CHUNK_SIZE=${SR_TRAIN_CHUNK_SIZE:-65536}
SR_EVAL_CHUNK_SIZE=${SR_EVAL_CHUNK_SIZE:-262144}

# 0 means every image in each split.
MAX_IMAGES_PER_SPLIT=${MAX_IMAGES_PER_SPLIT:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}
DRY_RUN=${DRY_RUN:-0}
SAVE_MODEL=${SAVE_MODEL:-0}
SAVE_PLOTS=${SAVE_PLOTS:-0}
LOG_IMAGE_EVOLUTION=${LOG_IMAGE_EVOLUTION:-0}

# Space-separated extra arguments. Use this only for simple values without
# embedded spaces, for example: EXTRA_ARGS="--mapping_size 256".
EXTRA_ARGS=${EXTRA_ARGS:-}

read -r -a gpu_array <<< "${GPU_IDS}"
read -r -a dataset_array <<< "${DATASETS}"
read -r -a model_array <<< "${MODELS}"
read -r -a optimizer_array <<< "${OPTIMIZERS}"
read -r -a epoch_array <<< "${EPOCHS}"
read -r -a scale_array <<< "${SCALE_FACTORS}"
read -r -a seed_array <<< "${SEEDS}"
read -r -a muon_lr_array <<< "${MUON_LRS}"
read -r -a rank_start_array <<< "${RANK_STARTS}"
read -r -a rank_end_array <<< "${RANK_ENDS}"
read -r -a fixed_rank_array <<< "${FIXED_RANKS}"
extra_arg_array=()
if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_arg_array <<< "${EXTRA_ARGS}"
fi

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( $1 > 0 ))
}

safe_tag() {
    local value=$1
    value=${value//./p}
    value=${value//\//-}
    value=${value// /-}
    printf '%s' "${value}"
}

canonical_optimizer() {
    case "$1" in
        auto-cos-inc|auto_cos_inc_rank|cosine_inc)       printf '%s' auto_cos_inc ;;
        auto-exp-inc|auto_exp_inc_rank|exp_inc)          printf '%s' auto_exp_inc ;;
        auto-log-inc|auto_log_inc_rank|log_inc)          printf '%s' auto_log_inc ;;
        auto-linear-inc|auto_linear_inc_rank|linear_inc) printf '%s' auto_linear_inc ;;
        auto-logistic-inc|auto_logistic_inc_rank|logistic_inc)
                                                          printf '%s' auto_logistic_inc ;;
        auto-cos-dec|auto_cos_dec_rank|cosine_dec)       printf '%s' auto_cos_dec ;;
        auto-fixed|auto_fixed_rank|fixed_rank)            printf '%s' auto_fixed ;;
        *)                                                printf '%s' "$1" ;;
    esac
}

learning_rate_for_model() {
    case "$1" in
        relu_mlp)  printf '%s' "${RELU_LR}" ;;
        siren_mlp) printf '%s' "${SIREN_LR}" ;;
        wire_mlp|real_wire)
                   printf '%s' "${WIRE_LR}" ;;
        finer_mlp) printf '%s' "${FINER_LR}" ;;
        *)         printf '%s' "${LR}" ;;
    esac
}

optimizer_shape_tag() {
    case "$1" in
        auto_cos_inc)      printf '%s' cos-inc ;;
        auto_exp_inc)      printf 'exp-%s' "$(safe_tag "${EXP_RATE}")" ;;
        auto_log_inc)      printf 'log-%s' "$(safe_tag "${LOG_RATE}")" ;;
        auto_linear_inc)   printf '%s' linear-inc ;;
        auto_logistic_inc) printf 'logistic-%s' "$(safe_tag "${LOGISTIC_STEEPNESS}")" ;;
        auto_cos_dec)      printf '%s' cos-dec ;;
        auto_fixed)        printf '%s' fixed ;;
    esac
}

resolve_scaled_step() {
    local configured_value=$1
    local ratio=$2
    local epochs=$3
    local setting_name=$4

    if [[ "${configured_value}" != auto ]]; then
        if ! is_positive_integer "${configured_value}"; then
            echo "[ERROR] ${setting_name} must be 'auto' or a positive integer; got '${configured_value}'." >&2
            return 2
        fi
        printf '%s' "${configured_value}"
        return 0
    fi

    if ! awk -v ratio="${ratio}" 'BEGIN { exit !(ratio > 0.0 && ratio <= 1.0) }'; then
        echo "[ERROR] ${setting_name} ratio must be in (0, 1]; got '${ratio}'." >&2
        return 2
    fi

    awk -v epochs="${epochs}" -v ratio="${ratio}" '
        BEGIN {
            step = int(epochs * ratio + 0.5)
            if (step < 1) step = 1
            if (step > epochs) step = epochs
            print step
        }
    '
}

rank_configurations_for_optimizer() {
    local optimizer=$1
    local low_rank
    local high_rank
    local fixed_rank
    local rank_floor

    if [[ "${optimizer}" == auto_fixed ]]; then
        for fixed_rank in "${fixed_rank_array[@]}"; do
            printf '%s\t%s\t%s\n' "${fixed_rank}" "${fixed_rank}" "${fixed_rank}"
        done
        return
    fi

    for low_rank in "${rank_start_array[@]}"; do
        rank_floor=${RANK_FLOOR:-${low_rank}}
        for high_rank in "${rank_end_array[@]}"; do
            if [[ "${optimizer}" == auto_cos_dec ]]; then
                printf '%s\t%s\t%s\n' "${rank_floor}" "${high_rank}" "${low_rank}"
            else
                printf '%s\t%s\t%s\n' "${rank_floor}" "${low_rank}" "${high_rank}"
            fi
        done
    done
}

dataset_root() {
    case "$1" in
        train) printf '%s' "${TRAIN_ROOT}" ;;
        valid) printf '%s' "${VALID_ROOT}" ;;
        *)
            echo "[ERROR] Unknown DATASETS label '$1'. Use train and/or valid." >&2
            return 2
            ;;
    esac
}

collect_images() {
    local root=$1
    local -n output_array=$2
    mapfile -d '' -t output_array < <(
        find "${root}" -maxdepth 1 -type f \
            \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
               -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.bmp' \) \
            -print0 | sort -z
    )
    if (( MAX_IMAGES_PER_SPLIT > 0 && ${#output_array[@]} > MAX_IMAGES_PER_SPLIT )); then
        output_array=("${output_array[@]:0:MAX_IMAGES_PER_SPLIT}")
    fi
}

append_status() {
    local timestamp=$1
    local status=$2
    local gpu_id=$3
    local job_id=$4
    local run_id=$5
    local exit_code=$6
    local run_dir=$7

    flock 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${timestamp}" "${status}" "${gpu_id}" "${job_id}" \
        "${run_id}" "${exit_code}" "${run_dir}" >&9
    flock -u 9
}

# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------
if [[ ! -d "${BASE_PATH}" ]]; then
    echo "[ERROR] BASE_PATH does not exist: ${BASE_PATH}" >&2
    exit 2
fi
if [[ ! -f "${FIT_SISR}" ]]; then
    echo "[ERROR] fit_sisr_rebut.py was not found: ${FIT_SISR}" >&2
    exit 2
fi
if [[ ! -f "${BASE_PATH}/rank_schedule_variants/__init__.py" ]]; then
    echo "[ERROR] rank_schedule_variants package was not found under ${BASE_PATH}." >&2
    exit 2
fi
if [[ ! -f "${BASE_PATH}/optims/rank_wsd_schedulers.py" ]]; then
    echo "[ERROR] rank_wsd scheduler was not found: ${BASE_PATH}/optims/rank_wsd_schedulers.py" >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable was not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] The 'flock' command is required for parallel status logging." >&2
    exit 2
fi
if [[ "${SCHEDULER}" != rank_wsd && "${SCHEDULER}" != rank-wsd ]]; then
    echo "[ERROR] This experiment script requires SCHEDULER=rank_wsd." >&2
    exit 2
fi
SCHEDULER=rank_wsd
if (( ${#gpu_array[@]} == 0 )); then
    echo "[ERROR] GPU_IDS must contain at least one physical GPU index." >&2
    exit 2
fi
if (( ${#model_array[@]} == 0 )); then
    echo "[ERROR] MODELS must contain at least one model name." >&2
    exit 2
fi
for model in "${model_array[@]}"; do
    case "${model}" in
        relu_mlp|siren_mlp|wire_mlp|real_wire|finer_mlp) ;;
        *)
            echo "[ERROR] Unsupported model '${model}'. Use relu_mlp, siren_mlp, wire_mlp, real_wire, and/or finer_mlp." >&2
            exit 2
            ;;
    esac
done
for gpu_id in "${gpu_array[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU index '${gpu_id}' in GPU_IDS." >&2
        exit 2
    fi
done
if [[ ! "${MAX_IMAGES_PER_SPLIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_IMAGES_PER_SPLIT must be zero or a positive integer." >&2
    exit 2
fi
if ! is_positive_integer "${MUON_NS_STEPS}"; then
    echo "[ERROR] MUON_NS_STEPS must be a positive integer." >&2
    exit 2
fi
if (( ${#muon_lr_array[@]} == 0 )); then
    echo "[ERROR] MUON_LRS must contain at least one learning rate." >&2
    exit 2
fi
for value in "${RELU_LR}" "${SIREN_LR}" "${WIRE_LR}" "${FINER_LR}" \
             "${muon_lr_array[@]}"; do
    if ! awk -v value="${value}" 'BEGIN { exit !(value > 0.0) }'; then
        echo "[ERROR] Learning rates must be positive numbers; got '${value}'." >&2
        exit 2
    fi
done

for value in "${epoch_array[@]}" "${scale_array[@]}" "${seed_array[@]}" \
             "${rank_start_array[@]}" "${rank_end_array[@]}" "${fixed_rank_array[@]}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Grid integer values must be non-negative integers; got '${value}'." >&2
        exit 2
    fi
done
for value in "${epoch_array[@]}" "${scale_array[@]}" \
             "${rank_start_array[@]}" "${rank_end_array[@]}" "${fixed_rank_array[@]}"; do
    if (( value < 1 )); then
        echo "[ERROR] Epoch, scale, and rank values must be positive; got '${value}'." >&2
        exit 2
    fi
done
if [[ -n "${RANK_FLOOR}" ]] && ! is_positive_integer "${RANK_FLOOR}"; then
    echo "[ERROR] RANK_FLOOR must be empty or a positive integer." >&2
    exit 2
fi
for low_rank in "${rank_start_array[@]}"; do
    for high_rank in "${rank_end_array[@]}"; do
        if (( high_rank < low_rank )); then
            echo "[ERROR] Every RANK_ENDS value must be >= every RANK_STARTS value." >&2
            exit 2
        fi
        if [[ -n "${RANK_FLOOR}" ]] && (( RANK_FLOOR > low_rank )); then
            echo "[ERROR] RANK_FLOOR must be <= each RANK_STARTS value." >&2
            exit 2
        fi
    done
done

mkdir -p "${OUT_ROOT}"
printf 'timestamp\tstatus\tgpu_id\tjob_id\trun_id\texit_code\trun_dir\n' > "${STATUS_FILE}"
exec 9>>"${STATUS_FILE}"

# ---------------------------------------------------------------------------
# Build a deterministic manifest covering both dataset splits
# ---------------------------------------------------------------------------
printf 'job_id\tdataset_split\timage_path\tmodel\toptimizer\trank_floor\trank_start\trank_end\tepochs\tscale_factor\tseed\tlr\tmuon_lr\trank_warmup_steps\tdecay_start_step\tcondition\trun_id\trun_dir\n' \
    > "${MANIFEST_FILE}"

job_count=0
for dataset_split in "${dataset_array[@]}"; do
    root=$(dataset_root "${dataset_split}")
    if [[ ! -d "${root}" ]]; then
        echo "[ERROR] Dataset directory does not exist: ${root}" >&2
        exit 2
    fi

    images=()
    collect_images "${root}" images
    if (( ${#images[@]} == 0 )); then
        echo "[ERROR] No supported image files were found in ${root}." >&2
        exit 2
    fi

    for model in "${model_array[@]}"; do
        model_lr=$(learning_rate_for_model "${model}")
        for optimizer_label in "${optimizer_array[@]}"; do
            optimizer=$(canonical_optimizer "${optimizer_label}")
            case "${optimizer}" in
                auto_cos_inc|auto_exp_inc|auto_log_inc|auto_linear_inc|auto_logistic_inc|auto_cos_dec|auto_fixed) ;;
                *)
                    echo "[ERROR] Unsupported rank_schedule_variants optimizer: ${optimizer_label}" >&2
                    exit 2
                    ;;
            esac

            mapfile -t rank_configs < <(rank_configurations_for_optimizer "${optimizer}")
            for rank_config in "${rank_configs[@]}"; do
                IFS=$'\t' read -r rank_floor rank_start rank_end <<< "${rank_config}"

                for epochs in "${epoch_array[@]}"; do
                    rank_warmup_steps=$(
                        resolve_scaled_step \
                            "${RANK_WARMUP_STEPS}" "${RANK_WARMUP_RATIO}" \
                            "${epochs}" RANK_WARMUP_STEPS
                    )
                    decay_start_step=$(
                        resolve_scaled_step \
                            "${RANK_WSD_DECAY_START_STEP}" "${RANK_WSD_DECAY_START_RATIO}" \
                            "${epochs}" RANK_WSD_DECAY_START_STEP
                    )

                    for scale_factor in "${scale_array[@]}"; do
                        for seed in "${seed_array[@]}"; do
                            for muon_lr in "${muon_lr_array[@]}"; do
                                condition=$(
                                    printf 'model-%s-L%s-H%s__opt-%s-shape-%s__sched-rank_wsd__x%s__rf%s-rs%s-re%s__rw%s-ds%s-min%s__ep%s__lr%s-mlr%s-mom%s-ns%s__seed%s' \
                                        "$(safe_tag "${model}")" "${NUM_LAYERS}" "${HIDDEN_DIM}" \
                                        "$(safe_tag "${optimizer}")" "$(optimizer_shape_tag "${optimizer}")" \
                                        "${scale_factor}" \
                                        "${rank_floor}" "${rank_start}" "${rank_end}" \
                                        "${rank_warmup_steps}" "${decay_start_step}" \
                                        "$(safe_tag "${RANK_WSD_MIN_LR_RATIO}")" "${epochs}" \
                                        "$(safe_tag "${model_lr}")" "$(safe_tag "${muon_lr}")" \
                                        "$(safe_tag "${MUON_MOMENTUM}")" "${MUON_NS_STEPS}" "${seed}"
                                )

                                for image_path in "${images[@]}"; do
                                    image_file=$(basename "${image_path}")
                                    image_stem=${image_file%.*}
                                    run_id="${EXPERIMENT_NAME}__${dataset_split}__${condition}__img-${image_stem}"
                                    run_dir="${OUT_ROOT}/runs/${dataset_split}/${condition}/${image_stem}"
                                    job_count=$((job_count + 1))

                                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                                        "${job_count}" "${dataset_split}" "${image_path}" "${model}" "${optimizer}" \
                                        "${rank_floor}" "${rank_start}" "${rank_end}" "${epochs}" \
                                        "${scale_factor}" "${seed}" "${model_lr}" "${muon_lr}" \
                                        "${rank_warmup_steps}" "${decay_start_step}" \
                                        "${condition}" "${run_id}" "${run_dir}" \
                                        >> "${MANIFEST_FILE}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

if (( job_count == 0 )); then
    echo "[ERROR] The generated manifest is empty." >&2
    exit 2
fi

echo "[INFO] Base path : ${BASE_PATH}"
echo "[INFO] Train data: ${TRAIN_ROOT}"
echo "[INFO] Valid data: ${VALID_ROOT}"
echo "[INFO] Output    : ${OUT_ROOT}"
echo "[INFO] GPUs      : ${GPU_IDS} (one sequential worker per GPU)"
echo "[INFO] Models    : ${MODELS}"
echo "[INFO] Muon LRs  : ${MUON_LRS}"
echo "[INFO] Model LRs : ReLU=${RELU_LR}, SIREN=${SIREN_LR}, WIRE=${WIRE_LR}, FINER=${FINER_LR}"
echo "[INFO] Jobs      : ${job_count}"
echo "[INFO] Manifest  : ${MANIFEST_FILE}"
echo "[INFO] Results   : ${RESULTS_CSV}"
echo "[INFO] Summary   : ${SUMMARY_CSV}"

# ---------------------------------------------------------------------------
# Worker: one sequential queue per physical GPU
# ---------------------------------------------------------------------------
run_worker() {
    local gpu_id=$1
    local worker_index=$2
    local gpu_count=$3
    local worker_completed=0
    local worker_failed=0
    local worker_skipped=0

    while IFS=$'\t' read -r job_id dataset_split image_path model optimizer \
        rank_floor rank_start rank_end epochs scale_factor seed lr muon_lr \
        rank_warmup_steps decay_start_step condition run_id run_dir; do

        if [[ "${job_id}" == job_id ]]; then
            continue
        fi
        if (( (job_id - 1) % gpu_count != worker_index )); then
            continue
        fi

        local done_marker="${run_dir}/.done"
        local failed_marker="${run_dir}/.failed"
        local running_marker="${run_dir}/.running"
        local log_file="${run_dir}/train.log"
        local command_file="${run_dir}/command.sh"

        mkdir -p "${run_dir}"
        if [[ "${SKIP_COMPLETED}" == 1 && -f "${done_marker}" ]]; then
            worker_skipped=$((worker_skipped + 1))
            append_status "$(date --iso-8601=seconds)" skipped "${gpu_id}" \
                "${job_id}" "${run_id}" 0 "${run_dir}"
            continue
        fi

        cmd=(
            "${PYTHON_BIN}" "${FIT_SISR}"
            --image "${image_path}"
            --task super_resolution
            --scale_factor "${scale_factor}"
            --optimizer "${optimizer}"
            --scheduler rank_wsd
            --epochs "${epochs}"
            --rank "${rank_floor}"
            --rank_start "${rank_start}"
            --rank_end "${rank_end}"
            --rank_warmup_steps "${rank_warmup_steps}"
            --rank_wsd_warmup_steps "${RANK_WSD_WARMUP_STEPS}"
            --rank_wsd_decay_start_step "${decay_start_step}"
            --rank_wsd_min_lr_ratio "${RANK_WSD_MIN_LR_RATIO}"
            --lr "${lr}"
            --muon_lr "${muon_lr}"
            --muon_momentum "${MUON_MOMENTUM}"
            --muon_ns_steps "${MUON_NS_STEPS}"
            --exp_rate "${EXP_RATE}"
            --log_rate "${LOG_RATE}"
            --logistic_steepness "${LOGISTIC_STEEPNESS}"
            --log_n_epochs "${LOG_N_EPOCHS}"
            --seed "${seed}"
            --model "${model}"
            --num_layers "${NUM_LAYERS}"
            --hidden_dim "${HIDDEN_DIM}"
            --sr_train_chunk_size "${SR_TRAIN_CHUNK_SIZE}"
            --sr_eval_chunk_size "${SR_EVAL_CHUNK_SIZE}"
            --experiment_name "${EXPERIMENT_NAME}"
            --dataset_split "${dataset_split}"
            --run_id "${run_id}"
            --output_dir "${run_dir}"
            --results_csv "${RESULTS_CSV}"
            --summary_csv "${SUMMARY_CSV}"
        )
        if [[ "${SAVE_MODEL}" == 1 ]]; then
            cmd+=(--save_model)
        fi
        if [[ "${SAVE_PLOTS}" != 1 ]]; then
            cmd+=(--skip_plots)
        fi
        if [[ "${LOG_IMAGE_EVOLUTION}" == 1 ]]; then
            cmd+=(--log_image_evolution)
        fi
        if (( ${#extra_arg_array[@]} > 0 )); then
            cmd+=("${extra_arg_array[@]}")
        fi

        {
            printf '#!/usr/bin/env bash\n'
            printf 'cd %q\n' "${BASE_PATH}"
            printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
            printf '%q ' "${cmd[@]}"
            printf '\n'
        } > "${command_file}"
        chmod +x "${command_file}"

        if [[ "${DRY_RUN}" == 1 ]]; then
            echo "[DRY-RUN][GPU ${gpu_id}][${job_id}/${job_count}] ${run_id}"
            printf '  CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
            printf '%q ' "${cmd[@]}"
            printf '\n'
            append_status "$(date --iso-8601=seconds)" dry_run "${gpu_id}" \
                "${job_id}" "${run_id}" 0 "${run_dir}"
            continue
        fi

        echo "[RUN][GPU ${gpu_id}][${job_id}/${job_count}] ${run_id}"
        rm -f "${done_marker}" "${failed_marker}"
        date --iso-8601=seconds > "${running_marker}"

        set +e
        (
            cd "${BASE_PATH}"
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
            "${cmd[@]}"
        ) > "${log_file}" 2>&1
        exit_code=$?
        set -e

        rm -f "${running_marker}"
        if (( exit_code == 0 )); then
            for required_output in predicted_image.png final_result.csv training_metrics.csv; do
                if [[ ! -f "${run_dir}/${required_output}" ]]; then
                    echo "[ERROR][GPU ${gpu_id}] Missing ${required_output}: ${run_dir}" >&2
                    exit_code=99
                    break
                fi
            done
        fi

        if (( exit_code == 0 )); then
            date --iso-8601=seconds > "${done_marker}"
            worker_completed=$((worker_completed + 1))
            append_status "$(date --iso-8601=seconds)" completed "${gpu_id}" \
                "${job_id}" "${run_id}" 0 "${run_dir}"
            echo "[DONE][GPU ${gpu_id}] ${run_id}"
        else
            printf '%s\n' "${exit_code}" > "${failed_marker}"
            worker_failed=$((worker_failed + 1))
            append_status "$(date --iso-8601=seconds)" failed "${gpu_id}" \
                "${job_id}" "${run_id}" "${exit_code}" "${run_dir}"
            echo "[FAILED][GPU ${gpu_id}] exit=${exit_code}: ${run_id}" >&2
            echo "                         log: ${log_file}" >&2
            if [[ "${STOP_ON_ERROR}" == 1 ]]; then
                return "${exit_code}"
            fi
        fi
    done < "${MANIFEST_FILE}"

    echo "[WORKER GPU ${gpu_id}] completed=${worker_completed} failed=${worker_failed} skipped=${worker_skipped}"
    (( worker_failed == 0 ))
}

worker_pids=()
gpu_count=${#gpu_array[@]}
for worker_index in "${!gpu_array[@]}"; do
    run_worker "${gpu_array[worker_index]}" "${worker_index}" "${gpu_count}" &
    worker_pids+=("$!")
done

worker_failures=0
set +e
for pid in "${worker_pids[@]}"; do
    wait "${pid}"
    worker_status=$?
    if (( worker_status != 0 )); then
        worker_failures=$((worker_failures + 1))
    fi
done
set -e

echo
if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[SUMMARY] Dry run completed; no training was launched."
else
    echo "[SUMMARY] All workers finished."
fi
echo "[SUMMARY] Per-image results : ${RESULTS_CSV}"
echo "[SUMMARY] Condition summary : ${SUMMARY_CSV}"
echo "[SUMMARY] Worker status     : ${STATUS_FILE}"
echo "[SUMMARY] Job manifest      : ${MANIFEST_FILE}"

if (( worker_failures > 0 )); then
    exit 1
fi