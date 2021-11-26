# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeRF',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=10, 
        # include_input=True),
        include_input=False),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=4, 
        # scale=128,
        # include_input=True),
        include_input=False),
    coarse_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*10,#+3,
        dir_emb_dims=2*3*4,#+3,
        use_dirs=True),
    fine_field=dict(
        type='BaseField',
        nb_layers=8, 
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
        # inv_depth=True,
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
            factor=8, 
            batch_size=None,
            split='train',
            batching=True, 
            # select_img=['005', '013', '015', '016'],
            select_img=['001', '005', '015', '019',],
            spherify=False, 
            no_ndc=False, 
            to_cuda=True,
            holdout=8),
        times=5),
    val=dict(
        type='LLFFDataset',
        datadir='~/data/3d/nerf/nerf_llff_data/fern', 
        factor=8, 
        batch_size=-1,
        split='val', 
        batching=False,
        # select_img=['014',],
        spherify=False, 
        no_ndc=False, 
        to_cuda=True,
        holdout=8),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
# optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999))
# optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**((1/1000)*(1/250)), by_epoch=False) 
# lr_config = dict(policy='Step', step=[40,80,120,160,180], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=200)
# lr_config = dict(policy='Step', step=[20,40,60,80,90], gamma=0.5, by_epoch=True)
# lr_config = dict(policy='Step', step=[50,80,90], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(policy='Poly', power=2, min_lr=5e-6, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
# lr_config = dict(policy='Step', step=[100,200,300], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=300)
# misc settings
checkpoint_config = dict(interval=5e3, by_epoch=False, max_keep_ckpts=5)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=5000),
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
        # inv_depth=True,
        use_dirs=True,
        max_rays_num=1024*2,))
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
