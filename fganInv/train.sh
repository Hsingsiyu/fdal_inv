cd   /home/xsy/invganV2/fganInv/ \
pip install lpips \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2   train.py  --gpu_id
s=[0,1] --divergence='pearson'  --configb

