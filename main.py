import os
from train import train
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def main():
    train()


if __name__ == '__main__':
    main()
