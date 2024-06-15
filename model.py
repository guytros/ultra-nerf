from run_nerf_helpers import *
from utils import batchify
import os


def run_network(inputs, fn, embed_fn, netchunk=512 * 32):
    """Prepares inputs and applies network 'fn'."""
    fn.run_eagerly = True
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk)(embedded)

    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, args.i_embed_gauss)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = args.output_ch
    skips = [4]

    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

    grad_vars = model.trainable_variables
    models = {'model': model}

    # We sample points equidistantly at the pixel location.
    # TODO: After sampling along a ray at any point use fine model
    # model_fine = None
    # if args.N_importance > 0:
    #     model_fine = init_nerf_model(
    #         D=args.netdepth_fine, W=args.netwidth_fine,
    #         input_ch=input_ch, output_ch=output_ch, skips=skips,
    #         input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    #     grad_vars += model_fine.trainable_variables
    #     models['model_fine'] = model_fine

    def network_query_fn(inputs, network_fn):
        return run_network(
            inputs, network_fn,
            embed_fn=embed_fn,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'network_fn': model
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)
        #
        # if model_fine is not None:
        #     ft_weights_fine = '{}_fine_{}'.format(
        #         ft_weights[:-11], ft_weights[-10:])
        #     print('Reloading fine from', ft_weights_fine)
        #     model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        B = self.kwargs['B']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if B is not None:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq, B=B: p_fn(x @ tf.transpose(B) * freq))
                    out_dim += d
                    out_dim += B.shape[1]
                else:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                            freq=freq,: p_fn(x * freq))
                    out_dim += d


        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, b=0):

    if i == -1:
        return tf.identity, 3
    if b != 0:
        #TODO: check seed
        B = tf.random.normal((b, 3), seed=1)
    else:
        B = None

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
        'B': B
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Model architecture

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=6, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W,
                                                         activation=act)
    #                                                     bias_initializer=tf.keras.initializers.RandomNormal(mean=-0.0, stddev=1.),
    #                                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=-0.0, stddev=1.))

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    print("input {}".format(inputs_pts.shape))
    for i in range(D):
        outputs = dense(W)(outputs)
        print("{} layer, {} shape".format(i, outputs.shape))
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)


    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)
        outputs = inputs_viewdirs
        for i in range(4):
            outputs = dense(W//2)(outputs)
        outputs = dense(output_ch-1, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)

    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
