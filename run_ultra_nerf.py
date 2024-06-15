import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import imageio
import time
import tensorflow_probability as tfp
from run_nerf_helpers import *
from load_us import load_us_data
from us_utilities import metrics
from tensorflow.keras import backend as K
from tqdm import tqdm
from render import gaussian_kernel
tf.compat.v1.enable_eager_execution()

from model import create_nerf
from train import train

# TODO: Change to args
g_size = 3
g_mean = 0.
g_variance = 1.
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = tf.constant(g_kernel[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)

def main():
    train()

if __name__ == '__main__':
    main()
