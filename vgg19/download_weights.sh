DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

wget --no-check-certificate 'https://s3-eu-west-1.amazonaws.com/explee-deep-learning/vgg19_weights.h5' -O $DIR/vgg-19_weights.hdf5
KERAS_BACKEND="theano" python $DIR/dump_headless_weights.py
python $DIR/dump_headless_weights.py
