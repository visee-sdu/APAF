DATA_ROOT=./datasets

train_alg=dagger

features=clip.b16
ft_dim=512
obj_features=vitbase
obj_ft_dim=0

ngpus=4
bs=4

object_dim=512
name=xxx

outdir=${DATA_ROOT}/R4R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r4r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy ndtw
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 200

      --batch_size ${bs}
      --lr 1e-5
      --iters 200000
      --log_every 500
      --aug_times 9

      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.8

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      --object_file ./object_feature/hm3d_mp3d_object_features.hdf5
      --object_dim ${object_dim}"


# train
# python -m torch.distributed.run --master_port $1 --nproc_per_node=$ngpus \
#       map_nav_src/r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --bert_ckpt_file use the path of the pretrained model \
#       --eval_first \
      

# test
python -m torch.distributed.run --master_port $1 --nproc_per_node=$ngpus \
      map_nav_src/r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file use the path of the trained model \
      --test --submit
