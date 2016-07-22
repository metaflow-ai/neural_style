import os, time, argparse

from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import Adam

# from keras.utils.visualize_util import plot as plot_model

from vgg19.model_headless import VGG_19_headless_4, get_layer_data

from utils.imutils import load_mean, get_image_list, load_image
from utils.lossutils import (grams, grams_output_shape, total_variation_error_keras)
from utils.general import mask_data, generate_data_from_image_list, import_model, get_shape
from utils.callbacks import TensorBoardBatch, ModelCheckpointBatch, HistoryBatch
from models.layers import custom_objects

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
results_dir = dir + '/models/data/st'
if not os.path.isdir(results_dir): 
    os.makedirs(results_dir)
train_dir = dataDir + '/train'
val_dir = dataDir + '/val'
overfit_dir = dataDir + '/overfit'
paintings_dir = dataDir + '/paintings'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--model_dir', default='', type=str, help='Load pretrained model')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--batch_size', default=4, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
parser.add_argument('--nb_epoch', default=2, type=int, help='Number of epoch.')
parser.add_argument('--max_epoch_size', default=-1, type=int, help='Max number of file to train on')
args = parser.parse_args()

dim_ordering = K.image_dim_ordering()
channels = 3
width = args.image_size
height = args.image_size
size = (height, width)
if dim_ordering == 'th':
    input_shape = (channels, width, height)
else:
    input_shape = (width, height, channels)
batch_size = args.batch_size

prefixed_dir = "%s/%s-%s-%s" % (results_dir, str(int(time.time())), K._BACKEND, dim_ordering)

if os.path.isdir(args.model_dir): 
    print("Loading pretrained model in %s" % (args.model_dir))
    st_model = import_model(args.model_dir, True, 
                should_convert=False, custom_objects=custom_objects)
else:
    raise Exception('You must provide a valid model_dir')

callbacks = []
if K._BACKEND == "tensorflow":
    tb = TensorBoardBatch(st_model, log_dir=prefixed_dir, histogram_freq=200, image_freq=51, write_graph=True)
    # We separate the image_freq and the histogram_freq just to avoid dumping everything at the same time
    callbacks.append(tb)
callbacks.append(
    ModelCheckpointBatch(st_model, chkp_dir=prefixed_dir + '/chkp', nb_step_chkp=200)
)
callbacks.append(
    HistoryBatch()
)

print('Loading VGG headless 4')
modelWeights = "%s/%s-%s-%s%s" % (vgg19Dir,'vgg-19', dim_ordering, K._BACKEND, '_headless_4_weights.hdf5')
vgg_model = VGG_19_headless_4(input_shape, modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(vgg_model, 'conv_')
style_layers = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
style_layers_mask = [name in style_layers for name in layers_names]
# About that: each time you have a pooling you hardly limiting the gradient flow upward
# after the > 2 layers, if you have a to do mini batchs and have a lot of noise
# You won't be able to pick up the gradient to converge
content_layers = ['conv_2_2']
content_layers_mask = [name in content_layers for name in layers_names]

print('Building full model')
mean = load_mean(name='vgg19') # BGR
if K._BACKEND == "tensorflow":
    import tensorflow as tf
    preprocessed_output = Lambda(lambda x: tf.reverse(x, [False, False, False, True]) - mean, name="ltv")(st_model.output) # RGB -> BGR -> BGR - mean
else:
    import theano.tensor as T
    preprocessed_output = Lambda(lambda x: T.reverse(x, [False, True, False, False]) - mean, name="ltv")(st_model.output) # RGB -> BGR -> BGR - mean
preds = vgg_model(preprocessed_output)
style_preds = mask_data(preds, style_layers_mask)
style_preds = [Lambda(grams, grams_output_shape, name='style' + str(idx))(style_pred) for idx, style_pred in enumerate(style_preds)]
content_preds = mask_data(preds, content_layers_mask)
full_model = Model(input=[st_model.input], output=content_preds + style_preds + [preprocessed_output])

# plot_model(full_model, to_file=results_dir + '/full_model.png', show_shapes=True)
# plot_model(vgg_model, to_file=results_dir + '/vgg_model.png', show_shapes=True)

print('Loading the generator')
tmp_vgg = VGG_19_headless_4(input_shape, modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(tmp_vgg, 'conv_')
content_output_layers = [layer_dict[lc_name].output for lc_name in content_layers]
true_content_f = K.function([tmp_vgg.input], content_output_layers)

style_fullpath_prefix = paintings_dir + '/results/' + args.style.split('/')[-1].split('.')[0]
train_image_list = get_image_list(train_dir)
if args.max_epoch_size > 0:
    train_image_list =train_image_list[:args.max_epoch_size]
train_samples_per_epoch = len(train_image_list)
train_generator = generate_data_from_image_list(
    train_image_list, (height, width), style_fullpath_prefix,
    input_len=1, output_len=6,
    batch_size=args.batch_size, transform_f=true_content_f, preprocess_type='none',
    verbose=False
)
val_image_list = get_image_list(val_dir)
if len(val_image_list) > 10:
    val_image_list = val_image_list[:10]
nb_val_samples = len(val_image_list)
val_generator = generate_data_from_image_list(
    val_image_list, (height, width), style_fullpath_prefix,
    input_len=1, output_len=6,
    batch_size=args.batch_size, transform_f=true_content_f, preprocess_type='none',
    verbose=False
)
# Tensorboard callback doesn't handle generator so far and we actually only need
# a few images to see the qualitative result
validation_data = []
validation_data.append(load_image(train_image_list[0], size=(height, width), preprocess_type='none'))
validation_data.append(load_image(val_image_list[0], size=(height, width), preprocess_type='none'))
st_model.validation_data = [validation_data]

print('Iterating over hyper parameters')
current_iter = 0
# Alpha need to be a lot lower than in the gatys_paper
# This is probably due to the fact that here we are looking at a new content picture each batch
# while the style is always the same and so he can go down the style gradient much faster than the content one
# which is  noisier
for alpha in [1e1]: 
    for beta in [1.]:
        for gamma in [1e-5]:
            print('Compiling model')
            adam = Adam(lr=1e-03, clipnorm=5.) # Clipping the norm avoid gradient explosion, no needs to suffocate it
            full_model.compile(optimizer=adam,
                loss=[
                    # content
                    'mse',
                    # style
                    'mse',
                    'mse',
                    'mse',
                    'mse',
                    # ltv
                    total_variation_error_keras
                ],
                loss_weights=[
                    # content
                    beta,
                    # style
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    # ltv
                    gamma
                ]
            )
            print('Training model')
            # This will set st_model for all callbacks even default those that i don't want ...
            # setattr(full_model, 'callback_model', st_model)
            # So i prefer to override the _set_model func
            base_history = full_model.fit_generator(train_generator, 
                samples_per_epoch=train_samples_per_epoch, 
                max_q_size=args.batch_size, 
                nb_epoch=args.nb_epoch, 
                verbose=1,
                callbacks=callbacks,
                validation_data=val_generator, 
                nb_val_samples=nb_val_samples,
            )
