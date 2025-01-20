conda activate paper0

python3 train_proposed.py \
    --name fine_corner_warp \
    --mode train \
    \
    --trainfrac 0.2 \
    --valfrac 0.5 \
    \
    --epochs 5 \
    --workers 8 \
    --batch 2 \
    --modeldepth 50 \
    \
    --baseH 100 \
    --baseW 100 \
    \
    --outH 300 \
    --outW 300 \
    \
    --scale 0 \
    --numscales 3 \
    --scaling 1.5 \
    \
    --strategy corner \
    \
    --shareencoder \
    --wconf 0.01 \
    \
    --preservecamera \
    \
    --maxsamplesperimage 8

python3 train_proposed.py \
    --name fine_grid_warp \
    --mode train \
    \
    --trainfrac 0.2 \
    --valfrac 0.5 \
    \
    --epochs 5 \
    --workers 8 \
    --batch 2 \
    --modeldepth 50 \
    \
    --outH 300 \
    --outW 300 \
    \
    --baseH 100 \
    --baseW 100 \
    \
    --scale 0 \
    --numscales 3 \
    --scaling 1.5 \
    \
    --strategy grid \
    \
    --shareencoder \
    --wconf 0.01 \
    \
    --preservecamera \
    \
    --maxsamplesperimage 8

python3 train_proposed.py \
    --name fine_random_warp \
    --mode train \
    \
    --trainfrac 0.2 \
    --valfrac 0.5 \
    \
    --epochs 5 \
    --workers 8 \
    --batch 2 \
    --modeldepth 50 \
    \
    --outH 300 \
    --outW 300 \
    \
    --baseH 100 \
    --baseW 100 \
    \
    --scale 0 \
    --numscales 3 \
    --scaling 1.5 \
    \
    --strategy random \
    \
    --shareencoder \
    --wconf 0.01 \
    \
    --preservecamera \
    \
    --maxsamplesperimage 8