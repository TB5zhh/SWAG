python -u -m indoor_scenes.lib.train \
    --train-batch-size 96 --val-batch-size 4 --num-worker 4 --epochs 200 --arch vit_b16 --resolution 224 --lr 1e-5