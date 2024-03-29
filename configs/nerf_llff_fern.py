# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeRF',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=10, 
        include_input=False),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=4, 
        scale=64,
        include_input=False),
    coarse_field=dict(
        type='BaseField',
        num_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*10,#+3,
        dir_emb_dims=2*3*4,#+3,
        use_dirs=True),
    fine_field=dict(
        type='BaseField',
        num_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*10,#+3,
        dir_emb_dims=2*3*4,#+3,
        use_dirs=True),
    render_params=dict( # default render cfg; train cfg
        n_samples=64,
        n_importance=128,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        use_dirs=True,
        max_rays_num=1024*3,))

# dataset settings
data = dict(
    samples_per_gpu=1024*4,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='LLFFDataset',
            datadir='~/data/3d/nerf/nerf_llff_data/fern', 
            factor=4, 
            batch_size=None,
            split='train',
            batching=True, 
            spherify=False, 
            no_ndc=False, 
            to_cuda=True,
            holdout=8),
        times=1),
    val=dict(
        type='LLFFDataset',
        datadir='~/data/3d/nerf/nerf_llff_data/fern', 
        factor=4, 
        batch_size=-1,
        split='val', 
        batching=False,
        spherify=False, 
        no_ndc=False, 
        to_cuda=True,
        holdout=8),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Poly', power=1, min_lr=5e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=int(2e5))
# misc settings
checkpoint_config = dict(interval=5e3, by_epoch=False, max_keep_ckpts=5)
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    epoch_interval=1,
    iter_interval=5e3,
    render_params=dict(
        n_samples=64,
        n_importance=128,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        use_dirs=True,
        max_rays_num=1024*2,))
extra_hooks = [dict(type='IterAdjustHook',), ]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
