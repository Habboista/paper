conda activate paper0

python3 train_proposed.py \
    --name coarse_grid_warp \
    --mode full \
    \
    --epochs 40 \
    --workers 4 \
    --batch 8 \
    --modeldepth 50 \
    \
    --baseH 100 \
    --baseW 100 \
    \
    --outH 300 \
    --outW 300 \
    \
    --scale 2 \
    --numscales 3 \
    --scaling 1.5 \
    \
    --strategy grid \
    \
    --shareencoder \
    --wconf 0.0 \
    \
    --maxsamplesperimage 4 \
    \
    --preservecamera

#python3 train_proposed.py \
#    --name coarse_grid_warp \
#    --mode train \
#    \
#    --trainfrac 0.2 \
#    --valfrac 0.5 \
#    \
#    --epochs 10 \
#    --workers 4 \
#    --batch 8 \
#    --modeldepth 50 \
#    \
#    --outH 300 \
#    --outW 300 \
#    \
#    --baseH 100 \
#    --baseW 100 \
#    \
#    --scale 2 \
#    --numscales 3 \
#    --scaling 1.5 \
#    \
#    --strategy grid \
#    \
#    --shareencoder \
#    --wconf 0.01 \
#    \
#    --preservecamera \
#    \
#    --maxsamplesperimage 4
#
#python3 train_proposed.py \
#    --name coarse_random_warp \
#    --mode train \
#    \
#    --trainfrac 0.2 \
#    --valfrac 0.5 \
#    \
#    --epochs 10 \
#    --workers 4 \
#    --batch 8 \
#    --modeldepth 50 \
#    \
#    --outH 300 \
#    --outW 300 \
#    \
#    --baseH 100 \
#    --baseW 100 \
#    \
#    --scale 2 \
#    --numscales 3 \
#    --scaling 1.5 \
#    \
#    --strategy random \
#    \
#    --shareencoder \
#    --wconf 0.01 \
#    \
#    --preservecamera \
#    \
#    --maxsamplesperimage 4