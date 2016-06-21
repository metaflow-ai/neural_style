import os, json, time, argparse

from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import Adam

# from keras.utils.visualize_util import plot as plot_model

from vgg19.model_headless import VGG_19_headless_5, get_layer_data
from models.style_transfer import (style_transfer_conv_transpose
                                , style_transfer_conv_inception_3)

from utils.imutils import plot_losses, load_mean, get_image_list, load_images
from utils.lossutils import (grams, frobenius_error, 
                    train_weights, total_variation_error_keras)
from utils.general import export_model, mask_data, generate_data_from_image_list
from utils.callbacks import TensorBoardBatch, ModelCheckpointBatch, HistoryBatch


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
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--weights', default='', type=str, help='Load pretrained weights')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--batch_size', default=4, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
parser.add_argument('--nb_epoch', default=2, type=int, help='Number of epoch.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
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

print('Loading style_transfer model')
st_model = style_transfer_conv_inception_3(mode=2, input_shape=input_shape, nb_res_layer=args.nb_res_layer) # th ordering, BGR
if os.path.isfile(args.weights): 
    print("Loading weights")
    st_model.load_weights(args.weights)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
vgg_model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type, input_shape=input_shape)
layer_dict, layers_names = get_layer_data(vgg_model, 'conv_')
style_layers = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
style_layers_mask = [name in style_layers for name in layers_names]
content_layers = ['conv_3_2']
content_layers_mask = [name in content_layers for name in layers_names]

print('Building full model')
mean = load_mean(name='vgg19', dim_ordering=dim_ordering) # th ordering, BGR
preprocessed_output = Lambda(lambda x: x - mean, name="ltv")(st_model.output) # th, BGR ordering, sub mean
preds = vgg_model(preprocessed_output)
style_preds = mask_data(preds, style_layers_mask)
style_preds = [Lambda(lambda x: grams(x), lambda shape: (shape[0], shape[1], shape[1]), name='style' + str(idx))(style_pred) for idx, style_pred in enumerate(style_preds)]
content_preds = mask_data(preds, content_layers_mask)
full_model = Model(input=[st_model.input], output=content_preds + style_preds + [preprocessed_output])

# plot_model(full_model, to_file=results_dir + '/full_model.png', show_shapes=True)
# plot_model(vgg_model, to_file=results_dir + '/vgg_model.png', show_shapes=True)

print('Loading the generator')
tmp_vgg = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type, input_shape=input_shape)
layer_dict, layers_names = get_layer_data(tmp_vgg, 'conv_')
content_layers = ['conv_3_2']
content_output_layers = [layer_dict[lc_name].output for lc_name in content_layers]
true_content_f = K.function([tmp_vgg.input], content_output_layers)

style_fullpath_prefix = paintings_dir + '/results/' + args.style.split('/')[-1].split('.')[0]
train_image_list = get_image_list(train_dir)
train_samples_per_epoch = len(train_image_list)
train_generator = generate_data_from_image_list(
    train_image_list, (height, width), style_fullpath_prefix, true_content_f,
    dim_ordering=dim_ordering, verbose=False, st=True
)
val_image_list = get_image_list(val_dir)
if len(val_image_list) > 40:
    val_image_list = val_image_list[:40]
nb_val_samples = len(val_image_list)
val_generator = generate_data_from_image_list(
    val_image_list, (height, width), style_fullpath_prefix, true_content_f,
    dim_ordering=dim_ordering, verbose=False, st=True
)
# Hack for the tensorboard
st_model.validation_data = None

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [2e2]:
    for beta in [1.]:
        for gamma in [1e-4]:
            prefixed_dir = results_dir + '/' + str(int(time.time()))

            print('Compiling model')
            adam = Adam(lr=1e-03)
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
            callbacks = []
            if K._BACKEND == "tensorflow":
                callbacks.append(
                    TensorBoardBatch(st_model, log_dir=prefixed_dir + '/logs', histogram_freq=200)
                )
            callbacks.append(
                ModelCheckpointBatch(st_model, chkp_dir=prefixed_dir + '/chkp', nb_step_chkp=200)
            )
            callbacks.append(
                HistoryBatch()
            )
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
            history = callbacks[-1]

            print("Saving final data")
            with open(prefixed_dir + '/losses.json', 'w') as outfile:
                json.dump(history.history, outfile)  
            plot_losses(history.history, prefixed_dir)
