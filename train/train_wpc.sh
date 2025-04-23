 python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --split_num 5 \
 --video_length_read 8 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 8 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/8zhen' \
 --ckpt_path 'ckpts/wpc_experiment1' \
| tee logs/wpc_8.1zhen.log

python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --split_num 5 \
 --video_length_read 8 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/8zhen' \
 --ckpt_path 'ckpts/wpc_experiment2' \
| tee logs/wpc_8.2zhen.log

python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --split_num 5 \
 --video_length_read 8 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 8 \
 --train_batch_size 32 \
 --num_workers 8 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/8zhen' \
 --ckpt_path 'ckpts/wpc_experiment3' \
| tee logs/wpc_8.3zhen.log

python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --split_num 5 \
 --video_length_read 8 \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 8 \
 --train_batch_size 32 \
 --num_workers 6 \
 --gpu_ids 0,1 \
 --epochs 50 \
 --kypath '/root/autodl-fs/8zhen' \
 --ckpt_path 'ckpts/wpc_experiment4' \
| tee logs/wpc_8.4zhen.log