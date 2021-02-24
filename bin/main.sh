# This script reproduces the experimental results by running multiple jobs at once.
cd ../src

python data.py      # Generate preprocessed datasets in `data/preprocessed`.

GPU_LIST="0 1 2 3"  # List of GPUs to use in experiments.
WORKERS=2           # The number of jobs that each GPU handles.

echo -e "data\ttrn_avg\ttrn_std\ttest_avg\ttest_std"
python run.py --data "brain-tumor"             --lr 5e-3 --rank 2 --workers $WORKERS --gpus $GPU_LIST
python run.py --data "breast-cancer"           --lr 5e-3 --rank 2 --workers $WORKERS --gpus $GPU_LIST
python run.py --data "breast-cancer-wisconsin" --lr 1e-3 --rank 2 --workers $WORKERS --gpus $GPU_LIST
python run.py --data "diabetes"                --lr 5e-4 --rank 1 --workers $WORKERS --gpus $GPU_LIST
python run.py --data "heart-disease"           --lr 2e-4 --rank 2 --workers $WORKERS --gpus $GPU_LIST
python run.py --data "hepatitis"               --lr 1e-4 --rank 2 --workers $WORKERS --gpus $GPU_LIST
