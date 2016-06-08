import os

from keras import backend as K

from utils.imutils import load_images, save_image_st, save_image

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
weightsDir = dir + '/models/weights/st'
if not os.path.isdir(weightsDir): 
    os.makedirs(weightsDir)
outputDir = dataDir + '/output'
if not os.path.isdir(outputDir): 
    os.makedirs(outputDir)
overfitDir = dataDir + '/overfit'

def dump_img_st(data):
    current_iter = data['current_iter']
    # losses = data['losses']
    st_model = data['st_model']
    X_overfit = load_images(overfitDir, limit=1, size=(512, 512), dim_ordering='th', verbose=True, st=True)
    predict = K.function([st_model.input, K.learning_phase()], st_model.output)

    results = predict([X_overfit, True])

    prefix = str(current_iter).zfill(4)
    fullOutPath = outputDir + '/' + prefix + "_callback_" + str(current_iter) + ".png"
    save_image_st(fullOutPath, results[0])
