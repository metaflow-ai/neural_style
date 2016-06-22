wget --no-check-certificate 'https://docs.google.com/uc?export=download&confirm=USJG&id=0Bz7KyqmuGsilZ2RVeVhKY0FyRmc' -O vgg-19_weights.hdf5
KERAS_BACKEND="theano" python vgg19/dump_headless_weights.py
python vgg19/dump_headless_weights.py
