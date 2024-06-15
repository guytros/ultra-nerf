import os
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# TODO: Improve psf shape. Now, it is a 2D Gaussian kernel.
def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    delta_t = 1  # 9.197324e-01
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t
    d1 = tf.distributions.Normal(mean, std * 2.)
    d2 = tf.distributions.Normal(mean, std)
    vals_x = d1.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32) * delta_t)
    vals_y = d2.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32) * delta_t)

    gauss_kernel = tf.einsum('i,j->ij',
                             vals_x,
                             vals_y)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


# Ray helpers
def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = tf.range(-W/2, W/2, dtype=tf.float32) * sw
    y = tf.zeros_like(x)
    z = tf.zeros_like(x)

    origin_base = tf.stack([x, y, z], axis=1)
    origin_base_prim = origin_base[..., None, :]
    origin_rotated = R * origin_base_prim
    ray_o_r = tf.reduce_sum(origin_rotated, axis=-1)
    rays_o = ray_o_r + t

    dirs_base = tf.constant([0., 1., 0.])
    dirs_r = tf.linalg.matvec(R, dirs_base)
    rays_d = tf.broadcast_to(dirs_r, rays_o.shape)

    return rays_o, rays_d


def render_rays_us(ray_batch,
                   network_fn,
                   network_query_fn,
                   N_samples,
                   retraw=False,
                   lindisp=False,
                   args=None):
    """Volumetric rendering.

        Args:
          ray_batch: array of shape [batch_size, ...]. We define rays and do not sample.

        Returns:

        """

    def raw2outputs(raw, z_vals, rays_d, g_kernel):
        """Transforms model's predictions to semantically meaningful values.
        """
        ret = render_method_convolutional_ultrasound(raw, z_vals, args, g_kernel)
        return ret

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Points in space to evaluate model at.
    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * \
           z_vals[..., :, None]

    pts = step + origin

    # Evaluate model at each point.
    raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples, 5]
    ret = raw2outputs(
        raw, z_vals, rays_d, g_kernel)

    if retraw:
        ret['raw'] = raw

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def batchify_rays(rays_flat, c2w=None, chunk=32 * 256, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_us(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_us(H, W, sw, sh,
              chunk=1024 * 32, rays=None, c2w=None,
              near=0., far=55. * 0.001,
              **kwargs):
    """Render rays
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
                tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, c2w=c2w, chunk=chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)
    return all_ret


def render_method_convolutional_ultrasound(raw, z_vals, args, g_kernel):

    def raw2attenualtion(raw, dists):
        return tf.exp(-raw * dists)

    # Compute distance between points
    # In paper the points are sampled equidistantly
    dists = tf.math.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = tf.squeeze(dists)
    dists = tf.concat([dists, dists[:, -1, None]], axis=-1)
    # ATTENUATION
    # Predict attenuation coefficient for each sampled point. This value is positive.
    attenuation_coeff = tf.math.abs(raw[..., 0])
    attenuation = raw2attenualtion(attenuation_coeff, dists)
    # Compute total attenuation at each pixel location as exp{-sum[a_n*d_n]}
    attenuation_transmission = tf.math.cumprod(attenuation, axis=1, exclusive=True)
    # REFLECTION
    prob_border = tf.math.sigmoid(raw[..., 2])

    # Bernoulli distribution can be approximated by RelaxedBernoulli
    # temperature = 0.01
    # border_distribution = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=prob_border)
    # Note: Estimating a border explicitly is not necessary. I recommend experimenting with solely relying on
    # reflection coefficient for the geometry estimation
    border_distribution = tf.contrib.distributions.Bernoulli(probs=prob_border, dtype=tf.float32)
    border_indicator = tf.stop_gradient(border_distribution.sample(seed=0))
    # Predict reflection coefficient. This value is between (0, 1).
    reflection_coeff = tf.math.sigmoid(raw[..., 1])
    reflection_transmission = 1. - reflection_coeff * border_indicator
    reflection_transmission = tf.math.cumprod(reflection_transmission, axis=1, exclusive=True)
    reflection_transmission = tf.squeeze(reflection_transmission)
    border_convolution = tf.nn.conv2d(input=border_indicator[tf.newaxis, :, :, tf.newaxis], filter=g_kernel,
                                      strides=1, padding="SAME")
    border_convolution = tf.squeeze(border_convolution)

    # BACKSCATTERING
    # Scattering density coefficient can be either learned or constant for fully developed speckle
    density_coeff_value = tf.math.sigmoid(raw[..., 3])
    density_coeff = tf.ones_like(reflection_coeff) * density_coeff_value
    scatter_density_distibution = tfp.distributions.Bernoulli(probs=density_coeff, dtype=tf.float32)
    scatterers_density = scatter_density_distibution.sample()
    # Predict scattering amplitude
    amplitude = tf.math.sigmoid(raw[..., 4])
    # Compute scattering template
    scatterers_map = tf.math.multiply(scatterers_density, amplitude)
    psf_scatter = tf.nn.conv2d(input=scatterers_map[tf.newaxis, :, :, tf.newaxis], filter=g_kernel, strides=1,
                               padding="SAME")
    psf_scatter = tf.squeeze(psf_scatter)
    # Compute remaining intensity at a point n
    transmission = tf.math.multiply(attenuation_transmission, reflection_transmission)
    # Compute backscattering part of the final echo
    b = tf.math.multiply(transmission, psf_scatter)
    # Compute reflection part of the final echo
    r = tf.math.multiply(tf.math.multiply(transmission, reflection_coeff), border_convolution)
    # Compute the final echo
    # Note: log compression has not been used for the submission
    # if args.log_compression:
    #     compression_constant = 3.14  # TODO: should be calculated based on r_reflection_maximum
    #     log_compression = lambda x: tf.math.log(1. + compression_constant * x) * tf.math.log(
    #         1. + compression_constant)
    #     r = log_compression(r)
    intensity_map = b + r
    ret = {'intensity_map': intensity_map,
           'attenuation_coeff': attenuation_coeff,
           'reflection_coeff': reflection_coeff,
           'attenuation_transmission': attenuation_transmission,
           'reflection_transmission': reflection_transmission,
           'scatterers_density': scatterers_density,
           'scatterers_density_coeff': density_coeff,
           'scatter_amplitude': amplitude,
           'b': b,
           'r': r,
           "transmission": transmission}
    return ret

# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def define_image_grid_3D_np(x_size, y_size):
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing='ij')
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid
