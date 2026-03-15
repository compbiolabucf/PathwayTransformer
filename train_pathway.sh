#!/usr/bin/env bash

set -euo pipefail

pathway="${pathway:-hsa04012}"
echo "pathway: ${pathway}"
mode="${mode:-train}"
mode="$(printf '%s' "${mode}" | tr '[:upper:]' '[:lower:]')"
echo "mode: ${mode}"

if [[ "${mode}" != "train" && "${mode}" != "test" ]]; then
    echo "invalid mode: ${mode}" >&2
    echo "allowed values: train test" >&2
    exit 1
fi

dataset_variant="${dataset_variant:-}"
valid_dataset_variants=(
    "brca_data1_as"
    "brca_data1_crapa"
    "brca_data1_gene"
    "brca_data1_utrapa"
    "brca_data2"
    "brca_data4"
)

if [[ -z "${dataset_variant}" ]]; then
    echo "Select dataset variant:"
    select selected_variant in "${valid_dataset_variants[@]}"; do
        if [[ -n "${selected_variant}" ]]; then
            dataset_variant="${selected_variant}"
            break
        fi
        echo "Invalid selection. Choose one of the listed dataset variants."
    done
fi

if [[ ! " ${valid_dataset_variants[*]} " =~ [[:space:]]${dataset_variant}[[:space:]] ]]; then
    echo "invalid dataset_variant: ${dataset_variant}" >&2
    echo "allowed values: ${valid_dataset_variants[*]}" >&2
    exit 1
fi

dataset_root="${dataset_root:-processed_data/${dataset_variant}}"
source_dataset="${dataset_root}/${pathway}"
target_dataset="${dataset_root}/ogbg_mol_breast_cancer"
if [[ ! -d "${source_dataset}" ]]; then
    echo "dataset not found: ${source_dataset}" >&2
    exit 1
fi

rm -rf "${target_dataset}"
cp -r "${source_dataset}" "${target_dataset}"

exp_name="${exp_name:-${pathway}-save-emb}"
seed="${seed:-1}"
arch="${arch:---ffn_dim 512 --hidden_dim 512 --dropout_rate 0.1 --n_layers 2 --edge_type multi_hop --multi_hop_max_dist 5}"
batch_size="${batch_size:-1}"
epoch="${epoch:-100}"
peak_lr="${peak_lr:-2e-4}"
end_lr="${end_lr:-1e-9}"
accelerator="${accelerator:-auto}"
devices="${devices:-1}"
strategy="${strategy:-auto}"
precision="${precision:-16-mixed}"
n_heads="${n_heads:-16}"
early_stopping_patience="${early_stopping_patience:-5}"
early_stopping="${early_stopping:-True}"
early_stopping_lower="$(printf '%s' "${early_stopping}" | tr '[:upper:]' '[:lower:]')"

max_epochs=$((epoch + 1))
default_root_dir="exps/brca/${exp_name}/${seed}"
tmp_dir="tmp/${exp_name}"
mkdir -p "${default_root_dir}" "${tmp_dir}"

early_stopping_arg="--enable_early_stopping"
checkpoint_name="best-val-loss.ckpt"
if [[ "${early_stopping_lower}" == "false" ]]; then
    early_stopping_arg="--disable_early_stopping"
    checkpoint_name="last.ckpt"
fi

checkpoint_dir="${default_root_dir}/lightning_logs/checkpoints"

resolve_checkpoint_path() {
    local preferred_file="${checkpoint_dir}/best-val-loss.ckpt"
    local fallback_file="${checkpoint_dir}/last.ckpt"

    if [[ -e "${preferred_file}" ]]; then
        printf '%s\n' "${preferred_file}"
        return 0
    fi
    if [[ -e "${fallback_file}" ]]; then
        printf '%s\n' "${fallback_file}"
        return 0
    fi
    return 1
}

run_validation_and_test() {
    local checkpoint_file="$1"

    echo "loading checkpoint for validation: ${checkpoint_file}"
    python "updated_codes/PT_train_test.py" \
        --num_workers 4 \
        --seed "${seed}" \
        --batch_size "${batch_size}" \
        --dataset_name ogbg_mol_breast_cancer \
        --dataset_root "${dataset_root}" \
        --accelerator "${accelerator}" \
        --devices "${devices}" \
        --strategy "${strategy}" \
        --precision "${precision}" \
        --default_root_dir "${tmp_dir}" \
        --checkpoint_path "${checkpoint_file}" \
        --validate \
        --early_stopping_patience "${early_stopping_patience}" \
        ${early_stopping_arg} \
        --log_every_n_steps 100 \
        --num_heads "${n_heads}" \
        ${arch}

    echo "loading checkpoint for testing: ${checkpoint_file}"
    python "updated_codes/PT_train_test.py" \
        --num_workers 4 \
        --seed "${seed}" \
        --batch_size "${batch_size}" \
        --dataset_name ogbg_mol_breast_cancer \
        --dataset_root "${dataset_root}" \
        --accelerator "${accelerator}" \
        --devices "${devices}" \
        --strategy "${strategy}" \
        --precision "${precision}" \
        --default_root_dir "${tmp_dir}" \
        --checkpoint_path "${checkpoint_file}" \
        --test \
        --early_stopping_patience "${early_stopping_patience}" \
        ${early_stopping_arg} \
        --log_every_n_steps 100 \
        --num_heads "${n_heads}" \
        ${arch}
}

if [[ "${mode}" == "train" ]]; then
    python "updated_codes/PT_train_test.py" \
        --num_workers 4 \
        --seed "${seed}" \
        --batch_size "${batch_size}" \
        --dataset_name ogbg_mol_breast_cancer \
        --dataset_root "${dataset_root}" \
        --accelerator "${accelerator}" \
        --devices "${devices}" \
        --strategy "${strategy}" \
        --precision "${precision}" \
        --default_root_dir "${default_root_dir}" \
        --max_epochs "${max_epochs}" \
        --early_stopping_patience "${early_stopping_patience}" \
        ${early_stopping_arg} \
        --num_heads "${n_heads}" \
        --peak_lr "${peak_lr}" \
        --end_lr "${end_lr}" \
        --log_every_n_steps 10 \
        ${arch}

    checkpoint_file="${checkpoint_dir}/${checkpoint_name}"
    if [[ -e "${checkpoint_file}" ]]; then
        run_validation_and_test "${checkpoint_file}"
    else
        echo "train the model first for this pathway and input data"
    fi
else
    if checkpoint_file="$(resolve_checkpoint_path)"; then
        echo "loading checkpoint from: ${checkpoint_file}"
        run_validation_and_test "${checkpoint_file}"
    else
        echo "train the model first for this pathway and input data using train mode"
    fi
fi
