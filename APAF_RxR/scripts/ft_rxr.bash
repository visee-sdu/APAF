DATA_ROOT=datasets

train_alg=dagger

features=vitclip
ft_dim=512
obj_features=vitbase
obj_ft_dim=0

ngpus=4
object_dim=512
name=sample

outdir=snap_ft/rxr/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --tokenizer xlm 
          
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

      --batch_size 1
      --lr 1e-5
      --iters 200000
      --log_every 500
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.8
      
      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      --pc_order 1
      --object_dim ${object_dim}
      "

export PYTHONPATH=build:map_nav_src:$PYTHONPATH

# train
# python -m torch.distributed.run --nproc_per_node=$ngpus --master_port=$1 \
#        map_nav_src/r2r/main_nav.py $flag \
#        --bert_ckpt_file use the path of the pretrained model \
#        --aug ./datasets/RxR/annotations/rxr_marky_train_guide_enc_xlmr.jsonl \
#        --eval_first
       

# test
python -m torch.distributed.run --nproc_per_node=$ngpus --master_port=$1 \
      map_nav_src/r2r/main_nav.py $flag  \
      --resume_file use the path of the trained model \
      --test --submit
