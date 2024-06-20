import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import io
import jax.numpy as jnp
from typing import Optional
tf.compat.v1.enable_eager_execution()


def create_log(args: argparse.Namespace):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
        return basedir, expname


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


# Misc utils
def show_colorbar(image, cmap='rainbow'):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf


def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def hybrid_loss(target_image: np.ndarray,
                pred_image: np.ndarray,
                ssim_filter_size,
                ssim_weight: Optional[float],
                loss_type: str = 'ssim') -> float:

    l2_loss = img2mse(pred_image, target_image)
    ssim_loss = 1. - tf.image.ssim_multiscale(tf.expand_dims(tf.expand_dims(pred_image, 0), -1),
                                              tf.expand_dims(tf.expand_dims(target_image, 0), -1),
                                              max_val=1.0, filter_size=ssim_filter_size,
                                              filter_sigma=1.5, k1=0.01, k2=0.1)

    if loss_type == 'l2':
        ssim_weight = 0
    loss = {'l2': (1 if loss_type == 'l2' else 1-ssim_weight, l2_loss),
            'ssim': (0 if loss_type == 'l2' else ssim_weight, ssim_loss)}
    loss['total'] = (1, loss['l2'][0]*loss['l2'][1]+loss['ssim'][0]*loss['ssim'][1])
    return loss


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def save_weights(net, prefix, i, basedir, expname):
    path = os.path.join(
        basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
    np.save(path, net.get_weights())
    print('saved weights at', path)


def patch_l2(y, y_prim):
    patches_y = tf.image.extract_patches(y[tf.newaxis, :, :, tf.newaxis], [1, 8, 8, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
    patches_y_prim = tf.image.extract_patches(y_prim[tf.newaxis, :, :, tf.newaxis], [1, 8, 8, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
    patch_sum_y = tf.reduce_sum(patches_y, axis=3, keepdims=True)
    patch_sum_y_prim = tf.reduce_sum(patches_y_prim, axis=3, keepdims=True)

    img2mse(tf.squeeze(patch_sum_y), tf.squeeze(patch_sum_y_prim))


def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.
    Note: The weighting / masking term was necessary to avoid degenerate
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[ :-1, :-1]
    v01 = values[ :-1, 1:]
    v10 = values[ 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = jnp.abs(v00 - v01) + jnp.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    if weighting is not None:

        loss = loss * weighting
    return loss


def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    delta_t = 1  # 9.197324e-01
    x_cos = np.array(list(range(-size, size +1)), dtype=np.float32)
    x_cos *= delta_t

    y_modulation = tf.cos(x_cos * 2 *np.pi *8e6)
    d1 = tf.distributions.Normal(mean, std *3.)

    d2 = tf.distributions.Normal(mean, std)
    vals_x = d1.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32 ) *delta_t)
    vals_y = d2.prob(tf.range(start=-size, limit=(size + 1), dtype=tf.float32 ) *delta_t)

    gauss_kernel = tf.einsum('i,j->ij',
                             vals_x,
                             vals_y)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)