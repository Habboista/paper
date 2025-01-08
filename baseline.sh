conda activate paper0

#python3 train_baseline.py \
#    --name baseline_half \
#    --mode full \
#    --epochs 40 \
#    --batch 24 \
#    --workers 16 \
#    --modeldepth 50 \
#    --velo \
#    --project \
#    \
#    --half

python3 train_baseline.py \
    --name baseline \
    --mode full \
    --epochs 40 \
    --batch 4 \
    --workers 16 \
    --modeldepth 50 \
    --velo \
    --project