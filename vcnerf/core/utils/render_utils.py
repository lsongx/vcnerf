import torch
from mmcv.runner import auto_fp16, force_fp32


im2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10 * torch.log(x) / torch.log(torch.as_tensor([10], device=x.device))


def raw2outputs(alphas, colors, z_vals, rays_dir, alpha_noise_std, white_bkgd):
    def process_alphas(alphas, dists, act_fn=torch.nn.functional.relu): 
        return 1 - torch.exp(-act_fn(alphas) * dists)

    # Computes distances
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # the distance that starts from the last point is infinity.
    dists = torch.cat([
        dists, 
        2e10 * torch.ones(dists[..., :1].shape, device=dists.device)
    ], dim=-1)  # [B, n_samples]

    # Multiplies each distance by the norm of its ray direction 
    # to convert it to real world distance (accounts for non-unit ray directions).
    dists = dists * torch.norm(rays_dir[..., None, :], dim=-1)

    # [B, n_points, 1] -> [B, n_points]
    alphas = alphas.squeeze(-1)

    # Adds noise to model's predictions for density. Can be used to 
    # regularize network (prevents floater artifacts).
    noise = 0
    if alpha_noise_std > 0:
        noise = torch.randn(alphas.shape, device=alphas.device) * alpha_noise_std

    # Predicts density of point. Higher values imply
    # higher likelihood of being absorbed at this point.
    alphas = process_alphas(alphas + noise, dists)  # [B, n_points]

    # Compute weight for RGB of each sample.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # tf has an args: exclusive, but torch does not, so I have to do all these complicated staffs.
    # [B, n_points]    
    weights = alphas * torch.cumprod(
        torch.cat([torch.ones(tuple(alphas.shape[:-1]) + (1,), device=alphas.device), 
                   1 - alphas[..., :-1] + 1e-10], dim=-1),
        dim=-1
    )
    # Computed weighted color of each sample y
    color_map = torch.sum(weights[..., None] * colors, dim=-2)  # [B, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [B]

    # Disparity map is inverse depth.
    disp_map = 1 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map
    if white_bkgd:
        color_map = color_map + (1 - acc_map[..., None]) 

    outputs = {
        'alphas': alphas, 
        'weights': weights, 
        'color_map': color_map, 
        'depth_map': depth_map, 
        'disp_map': disp_map, 
        'acc_map': acc_map
    }
    return outputs


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, n_importance, deterministic=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nan
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # [B, len(bins)]

    # Take uniform samples
    if deterministic:
        t_vals = torch.linspace(0., 1., steps=n_importance, device=bins.device)
        t_vals = t_vals.expand(tuple(cdf.shape[:-1]) + (n_importance,))
    else:
        t_vals = torch.rand(tuple(cdf.shape[:-1]) + (n_importance,), device=bins.device)
    t_vals = t_vals.contiguous()

    # Invert CDF
    indices = torch.searchsorted(cdf.detach(), t_vals)
    lower = torch.max(torch.zeros_like(indices - 1), indices - 1)
    upper = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)
    indices_g = torch.stack([lower, upper], -1)  # [B, n_importance, 2]

    matched_shape = [indices_g.shape[0], indices_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_g)

    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (t_vals - cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples
