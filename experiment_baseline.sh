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
    --name baseline_only_gt \
    --mode full \
    --epochs 40 \
    --batch 8 \
    --workers 16 \
    --modeldepth 50