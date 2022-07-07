cd   /home/xsy/invganV2/fganInv/ \
pip install lpips \
pip install imgaug
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2   train.py     --configb

python -m pytorch_fid path/to/dataset1 path/to/dataset2

