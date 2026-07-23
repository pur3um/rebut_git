#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Kodak24 image-fitting experiment grid
#
# Default grid:
#   images      : kodim01.png ... kodim24.png
#   epochs      : 2000, 3000
#   optimizers  : auto_cos_inc, lr_inc, lr_sign
#   scheduler   : rank_wsd
#   rank        : floor=rank_start, start=64, end=256
#
# Typical usage:
#   bash run_kodak24_optimizer_rank_grid.sh
#
# Run only auto_cos_inc:
#   OPTIMIZERS="auto_cos_inc" bash run_kodak24_optimizer_rank_grid.sh
#
# Try several starting ranks:
#   RANK_STARTS="32 64 128" bash run_kodak24_optimizer_rank_grid.sh
#
# Preview commands without training:
#   DRY_RUN=1 bash run_kodak24_optimizer_rank_grid.sh
#
# Resume an interrupted grid:
#   Re-run the same command. Runs containing .done are skipped by default.
# ============================================================================

# ---------------------------------------------------------------------------
# Server paths and runtime
# ---------------------------------------------------------------------------
BASE_PATH=${BASE_PATH:-/home/greenx9/data/cnfr/pp_rebuttal/muon_inrs}
DATA_ROOT=${DATA_ROOT:-"${BASE_PATH}/data/kodak24"}
FIT_IMAGE=${FIT_IMAGE:-"${BASE_PATH}/fit_image.py"}
OUT_ROOT=${OUT_ROOT:-"${BASE_PATH}/results/kodak24_optimizer_rank_grid"}
PYTHON_BIN=${PYTHON_BIN:-python}

# Physical GPU index. Inside fit_image.py this GPU is visible as cuda:0.
GPU_ID=${GPU_ID:-5}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# W&B still stores its run files inside each condition directory.
# Override with WANDB_MODE=online when online synchronization is desired.
WANDB_MODE=${WANDB_MODE:-offline}
PROJECT_NAME=${PROJECT_NAME:-pp-rebuttal-kodak24}

# ---------------------------------------------------------------------------
# Experiment grid
#
# Add future optimizers by appending their names to OPTIMIZERS. The helper
# below translates the current user-facing underscore aliases:
#   lr_inc  -> lr-inc
#   lr_sign -> lr-sign
#   lr_svd  -> lr-svd
# ---------------------------------------------------------------------------
OPTIMIZERS=${OPTIMIZERS:-"auto_cos_inc lr_inc lr_sign"}
SCHEDULERS=${SCHEDULERS:-"rank_wsd"}
EPOCHS=${EPOCHS:-"2000 3000"}
RANK_STARTS=${RANK_STARTS:-"64"}
RANK_ENDS=${RANK_ENDS:-"256"}

# Empty means that --rank follows each --rank_start. Set RANK_FLOOR=64 to keep
# the floor fixed at 64 while varying RANK_STARTS.
RANK_FLOOR=${RANK_FLOOR:-}

RANK_WARMUP_STEPS=${RANK_WARMUP_STEPS:-4000}
RANK_WSD_DECAY_START_STEP=${RANK_WSD_DECAY_START_STEP:-4000}
RANK_WSD_MIN_LR_RATIO=${RANK_WSD_MIN_LR_RATIO:-0.1}
RANK_WSD_WARMUP_STEPS=${RANK_WSD_WARMUP_STEPS:-0}

LR=${LR:-0.03}
MUON_LR=${MUON_LR:-3e-3}
SEED=${SEED:-42}
LOG_N_EPOCHS=${LOG_N_EPOCHS:-500}

# Operational controls
MAX_IMAGES=${MAX_IMAGES:-24}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}

# Optional extra fit_image.py arguments separated by spaces, for example:
#   EXTRA_ARGS="--model relu_mlp --hidden_dim 300"
EXTRA_ARGS=${EXTRA_ARGS:-}

read -r -a optimizer_array <<< "${OPTIMIZERS}"
read -r -a scheduler_array <<< "${SCHEDULERS}"
read -r -a epoch_array <<< "${EPOCHS}"
read -r -a rank_start_array <<< "${RANK_STARTS}"
read -r -a rank_end_array <<< "${RANK_ENDS}"
extra_arg_array=()
if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_arg_array <<< "${EXTRA_ARGS}"
fi

to_cli_optimizer() {
    case "$1" in
        lr_inc)  printf '%s' 'lr-inc' ;;
        lr_sign) printf '%s' 'lr-sign' ;;
        lr_svd)  printf '%s' 'lr-svd' ;;
        *)       printf '%s' "$1" ;;
    esac
}

to_cli_scheduler() {
    case "$1" in
        rank-wsd) printf '%s' 'rank_wsd' ;;
        *)        printf '%s' "$1" ;;
    esac
}

safe_tag() {
    local value=$1
    value=${value//./p}
    value=${value//\//-}
    value=${value// /-}
    printf '%s' "${value}"
}

append_summary() {
    local timestamp=$1
    local status=$2
    local optimizer=$3
    local scheduler=$4
    local rank_floor=$5
    local rank_start=$6
    local rank_end=$7
    local epochs=$8
    local image=$9
    local exit_code=${10}
    local run_dir=${11}

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${timestamp}" "${status}" "${optimizer}" "${scheduler}" \
        "${rank_floor}" "${rank_start}" "${rank_end}" "${epochs}" \
        "${image}" "${exit_code}" "${run_dir}" >> "${SUMMARY_FILE}"
}

if [[ ! -d "${BASE_PATH}" ]]; then
    echo "[ERROR] BASE_PATH does not exist: ${BASE_PATH}" >&2
    exit 2
fi
if [[ ! -f "${FIT_IMAGE}" ]]; then
    echo "[ERROR] fit_image.py was not found: ${FIT_IMAGE}" >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable was not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if (( MAX_IMAGES < 1 || MAX_IMAGES > 24 )); then
    echo "[ERROR] MAX_IMAGES must be between 1 and 24; got ${MAX_IMAGES}." >&2
    exit 2
fi

for image_index in $(seq 1 "${MAX_IMAGES}"); do
    printf -v image_stem 'kodim%02d' "${image_index}"
    image_path="${DATA_ROOT}/${image_stem}.png"
    if [[ ! -f "${image_path}" ]]; then
        echo "[ERROR] Required Kodak image was not found: ${image_path}" >&2
        exit 2
    fi
done

for epochs in "${epoch_array[@]}"; do
    if (( RANK_WARMUP_STEPS > epochs )); then
        echo "[WARNING] epochs=${epochs} is smaller than rank_warmup_steps=${RANK_WARMUP_STEPS}."
        echo "          The scheduled rank will not reach rank_end during this run."
    fi
    if (( RANK_WSD_DECAY_START_STEP >= epochs )); then
        echo "[WARNING] epochs=${epochs} is not larger than rank_wsd_decay_start_step=${RANK_WSD_DECAY_START_STEP}."
        echo "          rank_wsd may clamp decay_start_step to the final training step."
    fi
done

mkdir -p "${OUT_ROOT}"
SUMMARY_FILE="${OUT_ROOT}/run_summary.tsv"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    printf 'timestamp\tstatus\toptimizer\tscheduler\trank_floor\trank_start\trank_end\tepochs\timage\texit_code\trun_dir\n' \
        > "${SUMMARY_FILE}"
fi

total_jobs=$(( \
    ${#optimizer_array[@]} \
    * ${#scheduler_array[@]} \
    * ${#epoch_array[@]} \
    * ${#rank_start_array[@]} \
    * ${#rank_end_array[@]} \
    * MAX_IMAGES \
))

completed=0
failed=0
skipped=0
launched=0

echo "[INFO] Base path : ${BASE_PATH}"
echo "[INFO] Data path : ${DATA_ROOT}/kodim01.png ... kodim$(printf '%02d' "${MAX_IMAGES}").png"
echo "[INFO] Output    : ${OUT_ROOT}"
echo "[INFO] GPU       : physical GPU ${GPU_ID} only"
echo "[INFO] Jobs      : ${total_jobs}"
echo "[INFO] W&B mode  : ${WANDB_MODE}"

for optimizer_label in "${optimizer_array[@]}"; do
    cli_optimizer=$(to_cli_optimizer "${optimizer_label}")

    for scheduler_label in "${scheduler_array[@]}"; do
        cli_scheduler=$(to_cli_scheduler "${scheduler_label}")

        for rank_start in "${rank_start_array[@]}"; do
            if [[ -n "${RANK_FLOOR}" ]]; then
                rank_floor=${RANK_FLOOR}
            else
                rank_floor=${rank_start}
            fi

            for rank_end in "${rank_end_array[@]}"; do
                if (( rank_end < rank_start )); then
                    echo "[ERROR] rank_end (${rank_end}) must be >= rank_start (${rank_start})." >&2
                    exit 2
                fi

                for epochs in "${epoch_array[@]}"; do
                    optimizer_tag=$(safe_tag "${optimizer_label}")
                    scheduler_tag=$(safe_tag "${scheduler_label}")
                    lr_tag=$(safe_tag "${LR}")
                    muon_lr_tag=$(safe_tag "${MUON_LR}")
                    min_lr_tag=$(safe_tag "${RANK_WSD_MIN_LR_RATIO}")

                    condition_name=$(
                        printf 'opt-%s__sched-%s__rf%s-rs%s-re%s__rw%s-ds%s-min%s__ep%s__lr%s-mlr%s__seed%s' \
                            "${optimizer_tag}" "${scheduler_tag}" \
                            "${rank_floor}" "${rank_start}" "${rank_end}" \
                            "${RANK_WARMUP_STEPS}" "${RANK_WSD_DECAY_START_STEP}" \
                            "${min_lr_tag}" "${epochs}" "${lr_tag}" \
                            "${muon_lr_tag}" "${SEED}"
                    )

                    for image_index in $(seq 1 "${MAX_IMAGES}"); do
                        printf -v image_stem 'kodim%02d' "${image_index}"
                        image_path="${DATA_ROOT}/${image_stem}.png"
                        run_dir="${OUT_ROOT}/${condition_name}/${image_stem}"
                        done_marker="${run_dir}/.done"
                        failed_marker="${run_dir}/.failed"
                        running_marker="${run_dir}/.running"
                        log_file="${run_dir}/train.log"
                        command_file="${run_dir}/command.sh"
                        final_metrics_file="${run_dir}/final_metrics.txt"

                        mkdir -p "${run_dir}"

                        if [[ "${SKIP_COMPLETED}" == "1" && -f "${done_marker}" ]]; then
                            ((skipped += 1))
                            echo "[SKIP] ${condition_name}/${image_stem}"
                            append_summary \
                                "$(date --iso-8601=seconds)" "skipped" \
                                "${optimizer_label}" "${scheduler_label}" \
                                "${rank_floor}" "${rank_start}" "${rank_end}" \
                                "${epochs}" "${image_stem}" "0" "${run_dir}"
                            continue
                        fi

                        cmd=(
                            "${PYTHON_BIN}" "${FIT_IMAGE}"
                            --image "${image_path}"
                            --optimizer "${cli_optimizer}"
                            --scheduler "${cli_scheduler}"
                            --rank "${rank_floor}"
                            --rank_start "${rank_start}"
                            --rank_end "${rank_end}"
                            --rank_warmup_steps "${RANK_WARMUP_STEPS}"
                            --epochs "${epochs}"
                            --lr "${LR}"
                            --muon_lr "${MUON_LR}"
                            --seed "${SEED}"
                            --log_n_epochs "${LOG_N_EPOCHS}"
                            --project_name "${PROJECT_NAME}"
                        )

                        if [[ "${cli_scheduler}" == "rank_wsd" ]]; then
                            cmd+=(
                                --rank_wsd_warmup_steps "${RANK_WSD_WARMUP_STEPS}"
                                --rank_wsd_decay_start_step "${RANK_WSD_DECAY_START_STEP}"
                                --rank_wsd_min_lr_ratio "${RANK_WSD_MIN_LR_RATIO}"
                            )
                        elif [[ "${cli_scheduler}" == "cosine" ]]; then
                            cmd+=(--T_max "${epochs}")
                        fi

                        if (( ${#extra_arg_array[@]} > 0 )); then
                            cmd+=("${extra_arg_array[@]}")
                        fi

                        wandb_run_name="${image_stem}-${optimizer_tag}-${scheduler_tag}-rs${rank_start}-re${rank_end}-ep${epochs}"
                        wandb_group="${optimizer_tag}-${scheduler_tag}-rs${rank_start}-re${rank_end}"

                        {
                            printf '#!/usr/bin/env bash\n'
                            printf 'cd %q\n' "${BASE_PATH}"
                            printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
                            printf 'WANDB_MODE=%q ' "${WANDB_MODE}"
                            printf 'WANDB_DIR=%q ' "${run_dir}"
                            printf 'WANDB_NAME=%q ' "${wandb_run_name}"
                            printf 'WANDB_RUN_GROUP=%q ' "${wandb_group}"
                            printf '%q ' "${cmd[@]}"
                            printf '\n'
                        } > "${command_file}"
                        chmod +x "${command_file}"

                        if [[ "${DRY_RUN}" == "1" ]]; then
                            echo "[DRY-RUN] ${condition_name}/${image_stem}"
                            printf '          '
                            printf '%q ' "${cmd[@]}"
                            printf '\n'
                            append_summary \
                                "$(date --iso-8601=seconds)" "dry_run" \
                                "${optimizer_label}" "${scheduler_label}" \
                                "${rank_floor}" "${rank_start}" "${rank_end}" \
                                "${epochs}" "${image_stem}" "0" "${run_dir}"
                            continue
                        fi

                        ((launched += 1))
                        date --iso-8601=seconds > "${running_marker}"
                        rm -f "${done_marker}" "${failed_marker}"

                        echo "[RUN ${launched}/${total_jobs}] ${condition_name}/${image_stem}"
                        echo "                         log: ${log_file}"

                        set +e
                        (
                            cd "${BASE_PATH}"
                            export WANDB_MODE="${WANDB_MODE}"
                            export WANDB_DIR="${run_dir}"
                            export WANDB_NAME="${wandb_run_name}"
                            export WANDB_RUN_GROUP="${wandb_group}"
                            export WANDB_JOB_TYPE="image-fitting"
                            export WANDB_TAGS="kodak24,${optimizer_tag},${scheduler_tag},rank-start-${rank_start},epochs-${epochs}"
                            "${cmd[@]}"
                        ) 2>&1 | tee "${log_file}"
                        pipeline_status=("${PIPESTATUS[@]}")
                        set -e

                        exit_code=${pipeline_status[0]}
                        if (( ${pipeline_status[1]} != 0 && exit_code == 0 )); then
                            exit_code=${pipeline_status[1]}
                        fi

                        rm -f "${running_marker}"

                        if (( exit_code == 0 )); then
                            date --iso-8601=seconds > "${done_marker}"
                            awk '/^Final Full Image (PSNR|SSIM|LPIPS):/' "${log_file}" \
                                > "${final_metrics_file}" || true
                            ((completed += 1))
                            echo "[DONE] ${condition_name}/${image_stem}"
                            append_summary \
                                "$(date --iso-8601=seconds)" "completed" \
                                "${optimizer_label}" "${scheduler_label}" \
                                "${rank_floor}" "${rank_start}" "${rank_end}" \
                                "${epochs}" "${image_stem}" "${exit_code}" "${run_dir}"
                        else
                            printf '%s\n' "${exit_code}" > "${failed_marker}"
                            ((failed += 1))
                            echo "[FAILED] exit=${exit_code}: ${condition_name}/${image_stem}" >&2
                            append_summary \
                                "$(date --iso-8601=seconds)" "failed" \
                                "${optimizer_label}" "${scheduler_label}" \
                                "${rank_floor}" "${rank_start}" "${rank_end}" \
                                "${epochs}" "${image_stem}" "${exit_code}" "${run_dir}"

                            if [[ "${STOP_ON_ERROR}" == "1" ]]; then
                                echo "[ERROR] STOP_ON_ERROR=1, stopping the grid." >&2
                                exit "${exit_code}"
                            fi
                        fi
                    done
                done
            done
        done
    done
done

echo
echo "[SUMMARY] completed=${completed} failed=${failed} skipped=${skipped}"
echo "[SUMMARY] table: ${SUMMARY_FILE}"

if (( failed > 0 )); then
    exit 1
fi