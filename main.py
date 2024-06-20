import os
import wandb
import numpy as np
import tensorflow as tf

from config import config_parser
from train import train
from inference import inference

tf.compat.v1.enable_eager_execution()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def main():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    if args.run_mode == 'train':
        wandb.init(
            entity='guytros',
            project=args.project_name,
            config=vars(args),
            name=args.expname,
            save_code=True,
        )
        train(args)
        wandb.finish()

    elif args.run_mode == 'inference':
        inference(args)


if __name__ == '__main__':
    main()
