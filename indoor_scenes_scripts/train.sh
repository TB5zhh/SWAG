python -m indoor_scenes.lib.train \
    --batch-size 64 --num-worker 2 --epochs 100 --arch vit_b16 --resolution 224 --lr 1e-5