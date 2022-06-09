# A PyTorch implementation of NeRF
- [x] FP16
- [x] distributed training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --use_fp16 1
```
