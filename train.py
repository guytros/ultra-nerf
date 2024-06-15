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
from utils import img2mse, show_colorbar
from tensorflow.keras import backend as K


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

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

    # The poses are not normalized. We scale down the space.
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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

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
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
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
        l2_weight = 1. - ssim_weight

        rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, pose)
        batch_rays = tf.stack([rays_o, rays_d], 0)
        loss = dict()
        loss_holdout = dict()
        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            # Make predictions
            rendering_output = render_us(
                H, W, sw, sh, c2w=pose, chunk=args.chunk, rays=batch_rays,
                retraw=True, **render_kwargs_train)

            output_image = rendering_output['intensity_map']
            if args.loss == 'l2':
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (1., l2_intensity_loss)
            elif args.loss == 'ssim':
                ssim_intensity_loss = 1. - tf.image.ssim_multiscale(tf.expand_dims(tf.expand_dims(output_image, 0), -1),
                                                                    tf.expand_dims(tf.expand_dims(target, 0), -1),
                                                                    max_val=1.0, filter_size=args.ssim_filter_size,
                                                                    filter_sigma=1.5, k1=0.01, k2=0.1
                                                                    )
                loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (l2_weight, l2_intensity_loss)

            total_loss = 0.
            for loss_value in loss.values():
                total_loss += loss_value[0] * loss_value[1]

        gradients = tape.gradient(total_loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))
        dt = time.time() - time0

        #####           end            #####

        # Rest is logging
        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, total_loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
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
                tf.contrib.summary.scalar('train/total_loss/', total_loss)
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
                if args.loss == 'l2':
                    l2_intensity_loss = img2mse(output_image_test, target)
                    loss_holdout["l2"] = (1., l2_intensity_loss)
                elif args.loss == 'ssim':
                    ssim_intensity_loss = 1. - tf.image.ssim_multiscale(
                        tf.expand_dims(tf.expand_dims(output_image_test, 0), -1),
                        tf.expand_dims(tf.expand_dims(target, 0), -1),
                        max_val=1.0, filter_size=args.ssim_filter_size,
                        filter_sigma=1.5, k1=0.01, k2=0.1
                    )
                    loss_holdout["ssim"] = (ssim_weight, ssim_intensity_loss)
                    l2_intensity_loss = img2mse(output_image_test, target)
                    loss_holdout["l2"] = (l2_weight, l2_intensity_loss)

                total_loss_holdout = 0.
                for loss_value in loss_holdout.values():
                    total_loss_holdout += loss_value[0] * loss_value[1]

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
                    tf.contrib.summary.scalar('test/total_loss/', total_loss)
                    tf.contrib.summary.image('b_mode/target/',
                                             tf.expand_dims(tf.expand_dims(to8b(tf.transpose(target)), 0), -1))
                    for map_k, map_v in rendering_output_test.items():
                        tf.contrib.summary.image(f'maps/{map_k}/',
                                                 tf.expand_dims(tf.image.decode_png(
                                                     show_colorbar(tf.transpose(map_v)).getvalue(), channels=4),
                                                     0))

        global_step.assign_add(1)
