#!/bin/bash

pathway=hsa04012
echo "pathway: ${pathway}"
if [ -d dataset/ogbg_mol_breast_cancer ]; then 
    rm -r dataset/ogbg_mol_breast_cancer
fi
cp -r dataset/${pathway} dataset/ogbg_mol_breast_cancer
exp_name="${pathway}-save-emb"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 512 --hidden_dim 512 --dropout_rate 0.1 --n_layers 2 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="1"       
[ -z "${epoch}" ] && epoch="8"  
[ -z "${peak_lr}" ] && peak_lr="2e-4"  
[ -z "${end_lr}" ] && end_lr="1e-9"
#[ -z "${flag_m}" ] && flag_m="2"
#[ -z "${flag_step_size}" ] && flag_step_size="0.2"
#[ -z "${flag_mag}" ] && flag_mag="0"

[ -z "${ckpt_path}" ] && ckpt_path=""

# echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "ckpt_path ${ckpt_path}"
echo "arch: ${arch}"
echo "batch_size: ${batch_size}"
echo "peak_lr ${peak_lr}"
echo "end_lr ${end_lr}"
#echo "flag_m ${flag_m}"
#echo "flag_step_size :${flag_step_size}"
#echo "flag_mag: ${flag_mag}"
echo "seed: ${seed}"
echo "epoch: ${epoch}"
# echo "==============================================================================="

n_gpu=1                 
#tot_updates=$((33000*epoch/batch_size/n_gpu))
#warmup_updates=$((tot_updates/10))
max_epochs=$((epoch+1))
# echo "=====================================ARGS======================================"
#echo "tot_updates ${tot_updates}"
#echo "warmup_updates: ${warmup_updates}"
echo "max_epochs: ${max_epochs}"
echo "==============================================================================="

default_root_dir=exps/brca/$exp_name/$seed
n_heads=16
mkdir -p $default_root_dir

tmp_dir=tmp/$exp_name
mkdir -p $tmp_dir

python main_codes/entry.py --num_workers 4 --seed $seed --batch_size $batch_size \
    --dataset_name ogbg_mol_breast_cancer \
    --gpus 1 --accelerator ddp --precision 16 $arch \
    --default_root_dir $default_root_dir \
    --max_epochs $max_epochs --num_heads $n_heads \
    --peak_lr $peak_lr --end_lr $end_lr --progress_bar_refresh_rate 10 \
    

checkpoint_dir=$default_root_dir/lightning_logs/checkpoints/
echo "=====================================EVAL======================================"
for file in `ls $checkpoint_dir/last.ckpt`
do
    echo -e "\n\n\n ckpt:"
    echo "$file"
    echo -e "\n\n\n"
    python main_codes/entry.py --num_workers 4 --seed $seed --batch_size $batch_size \
            --dataset_name ogbg_mol_breast_cancer \
            --gpus 1 --accelerator ddp --precision 16 $arch \
            --default_root_dir $tmp_dir/ \
            --checkpoint_path $file --validate --progress_bar_refresh_rate 100 --num_heads $n_heads

    python main_codes/entry.py --num_workers 4 --seed $seed --batch_size $batch_size \
            --dataset_name ogbg_mol_breast_cancer \
            --gpus 1 --accelerator ddp --precision 16 $arch \
            --default_root_dir $tmp_dir/ \
            --checkpoint_path $file --test --progress_bar_refresh_rate 100 --num_heads $n_heads
done
echo "==============================================================================="
