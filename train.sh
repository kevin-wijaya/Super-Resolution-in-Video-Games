CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp0.yml \
--launcher pytorch


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp1.yml \
--launcher pytorch

##########################################
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp2.yml \
--launcher pytorch

##########################################
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp8.yml \
--launcher pytorch

##########################################
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp9.yml \
--launcher pytorch

##########################################
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--use-env --nproc_per_node=2 --master_port=1146  basicsr/train.py \
-opt options/train/exp11.yml \
--launcher pytorch

##########################################
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py \
-opt options/train/exp12.yml \
--launcher pytorch
# rsync -avh \
#   --info=progress2 \
#   --exclude="experiments/*" \
#   --exclude="dataset/*" \
#   --exclude="results/*" \
#   ./ \
#   hungld@140.113.24.145:/mnt/HDD6/hungld/selected_topics/PFT-SR/


rsync -avh \
  --info=progress2 \
  ./dataset/train/ \
  hungld@140.113.24.145:/mnt/HDD6/hungld/selected_topics/PFT-SR/dataset/

rsync -avh \
  --info=progress2 \
  ./training_dataset_2 \
  hungld@140.113.24.148:/mnt/HDD4/hungld/select_topics/final/PFT-SR/