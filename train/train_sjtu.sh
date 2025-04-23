 python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --video_length_read 9 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 8 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/9zhen' \
 --ckpt_path 'ckpts/sjtu_experiment1' \
| tee log/sjtu_9.1zhen.log

 python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --video_length_read 9 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/9zhen' \
 --ckpt_path 'ckpts/sjtu_experiment2' \
| tee log/sjtu_9.2zhen.log

 python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --video_length_read 9 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 8 \
 --train_batch_size 32 \
 --num_workers 8 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/9zhen' \
 --ckpt_path 'ckpts/sjtu_experiment3' \
| tee log/sjtu_9.3zhen.log

python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --video_length_read 9 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 8 \
 --train_batch_size 32 \
 --num_workers 6 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/9zhen' \
 --ckpt_path 'ckpts/sjtu_experiment4' \
| tee log/sjtu_9.4zhen.log
