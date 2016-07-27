import os, argparse

from keras import backend as K
if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    saver = tf.train.Saver()
else:
    saver = None

from utils.general import export_model, import_model
from models.layers import custom_objects

dir = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--model_dir', type=str, help='Model absolute directory')
args = parser.parse_args()

args.model_dir
if not os.path.isdir(args.model_dir): 
    raise Exception("The model_dir is not a directory")
model = import_model(args.model_dir, should_convert=False, custom_objects=custom_objects)
print(model.input.name, model.output.name)

export_model(model, dir + "/../models/data/output", saver=saver)
