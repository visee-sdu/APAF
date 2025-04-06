
NODE_RANK=0
NUM_GPUS=4
outdir=../datasets/R2R/exprs_map/pretrain/r2r_b16

# train
python -m torch.distributed.launch \
    --master_port $1 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir 
