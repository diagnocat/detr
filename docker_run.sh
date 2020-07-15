#!/usr/bin/env bash

docker build -t detr .

# docker run -it --rm \
#     --gpus all \
#     --shm-size=8g \
#     --net host \
#     -v `pwd`:/workspace/detr \
#     -v $HOME/coco:/workspace/coco \
#     detr bash -c "cd /workspace/detr && jupyter lab --ip 0.0.0.0 --port 10803 --no-browser --allow-root"

docker run -it --rm \
    --gpus all \
    --shm-size=8g \
    --net host \
    -v $HOME/.torch:/root/.torch \
    -v $HOME/.cache/torch:/root/.cache/torch \
    -v /data/hdd1/dvc-cache:/dvc-cache \
    -v /data/hdd1/dvc-cache:/data/dvc-cache \
    -v `pwd`:/workspace/detr \
    -v /home/ematvey/dc/research/datasets/xray_v2/coco:/workspace/coco \
    detr bash -c "cd /workspace/detr && ipython --pdb -m torch.distributed.launch -- --nproc_per_node=4 --use_env main.py --coco_path /workspace/coco --output_dir output_1 --batch_size 2 --resume=/workspace/detr/output_1/checkpoint.pth"

# docker run -it --rm \
#     --gpus all \
#     --shm-size=8g \
#     --net host \
#     -v $HOME/.torch:/root/.torch \
#     -v $HOME/.cache/torch:/root/.cache/torch \
#     -v /data/hdd1/dvc-cache:/dvc-cache \
#     -v /data/hdd1/dvc-cache:/data/dvc-cache \
#     -v `pwd`:/workspace/detr \
#     -v /home/ematvey/dc/research/datasets/xray_v2/coco:/workspace/coco \
#     detr bash -c "cd /workspace/detr && ipython --pdb main.py -- --num_workers=0 --coco_path /workspace/coco"
