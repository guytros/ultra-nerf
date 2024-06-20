import argparse

import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import os
from config import config_parser
from load_us import load_us_data
from model import create_nerf
from render import get_rays_us_linear, render_us, to8b
from utils import img2mse, show_colorbar, create_log, hybrid_loss, save_weights
from tensorflow.keras import backend as K
import wandb


def train(args: argparse.Namespace):

    # Load data
    if args.dataset_type == 'us':
        images, poses, i_test = load_us_data(args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print("Test {}, train {}".format(len(i_test), len(i_train)))

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # The poses are not (!) normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth
    H, W = images.shape[1], images.shape[2]
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx
    # H, W = int(H), int(W)

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir, expname = create_log(args)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train["args"] = args

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    N_iters = args.n_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    # training loop
    for i in tqdm(range(start, N_iters)):

        time0 = time.time()
        # Sample random ray batch
        # Random from one image
        img_i = np.random.choice(i_train)
        try:
            target = tf.transpose(images[img_i])
        except:
            print(img_i)

        pose = poses[img_i, :3, :4]
        ssim_weight = args.ssim_lambda

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)
        batch_rays = tf.stack([rays_o, rays_d], 0)

        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            # Make predictions
            rendering_output = render_us(
                H, W, sw, sh, c2w=pose, chunk=args.chunk, rays=batch_rays,
                retraw=True, **render_kwargs_train)

            output_image = rendering_output['intensity_map']
            loss = hybrid_loss(target_image=target, pred_image=output_image, ssim_filter_size=args.ssim_filter_size,
                               ssim_weight=args.ssim_lambda, loss_type=args.loss)
            wandb.log({"train_L2": loss['l2'][1], "train_SSIM": loss['ssim'][1], "train_loss": loss['total'][1]}, step=i)

        gradients = tape.gradient(loss['total'][1], grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        batch_time = time.time() - time0

        #####           end            #####

        # Rest is logging

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i, basedir, expname)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, loss['total'][1].numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(batch_time))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                g_i = 0
                for t in gradients:
                    g_i += 1
                    tf.contrib.summary.histogram(str(g_i), t)
                tf.contrib.summary.scalar('misc/learning_rate', K.eval(optimizer.learning_rate(optimizer.iterations)))
                loss_string = "Total loss = "
                for l_key, l_value in loss.items():
                    loss_string += f' + {l_value[0]} * {l_key}'
                    tf.contrib.summary.scalar(f'train/loss_{l_key}/', l_value[1])
                    tf.contrib.summary.scalar(f'train/penalty_factor_{l_key}/', l_value[0])
                    tf.contrib.summary.scalar(f'train/total_loss_{l_key}/', l_value[0] * l_value[1])
                tf.contrib.summary.scalar('train/total_loss/', loss['total'][1])
                print(loss_string)
            if i % args.i_img == 0:
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = tf.transpose(images[img_i])
                pose = poses[img_i, :3, :4]
                rendering_output_test = render_us(H, W, sw, sh, chunk=args.chunk, c2w=pose,
                                                  **render_kwargs_test)

                # TODO: Duplicaetes the loss calculation. Should be a function.
                output_image_test = rendering_output_test['intensity_map']
                loss_holdout = hybrid_loss(target_image=target, pred_image=output_image_test, ssim_filter_size=args.ssim_filter_size,
                                           ssim_weight=args.ssim_lambda, loss_type=args.loss)
                wandb.log({"infer_L2": loss_holdout['l2'][1], "infer_SSIM": loss_holdout['ssim'][1], "infer_loss": loss_holdout['total'][1]},
                          step=i)
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                # if i==0:
                os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir,
                                             '{:06d}.png'.format(i)), to8b(tf.transpose(output_image_test)))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('b_mode/output/',
                                             tf.expand_dims(tf.expand_dims(to8b(tf.transpose(output_image_test)), 0),
                                                            -1))
                    for l_key, l_value in loss_holdout.items():
                        tf.contrib.summary.scalar(f'test/loss_{l_key}/', l_value[0])
                        tf.contrib.summary.scalar(f'test/penalty_factor_{l_key}/', l_value[1])
                        tf.contrib.summary.scalar(f'test/total_loss_{l_key}/', l_value[0] * l_value[1])
                    tf.contrib.summary.scalar('test/total_loss/', loss_holdout['total'][1])
                    tf.contrib.summary.image('b_mode/target/',
                                             tf.expand_dims(tf.expand_dims(to8b(tf.transpose(target)), 0), -1))
                    for map_k, map_v in rendering_output_test.items():
                        tf.contrib.summary.image(f'maps/{map_k}/',
                                                 tf.expand_dims(tf.image.decode_png(
                                                     show_colorbar(tf.transpose(map_v)).getvalue(), channels=4),
                                                     0))

        global_step.assign_add(1)
