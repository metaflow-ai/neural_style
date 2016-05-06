import numpy as np
import os, time
import png

from models.style_transfer import style_transfer
from utils.imutils import load_images

dir = os.path.dirname(os.path.realpath(__file__))

print('Loading test set')
# X_test = load_images(dir + '/data/test')
X_test = load_images(dir + '/data/overfit')
print(X_test.shape)

print('Loading style_transfer')
stWeights = dir + '/models/st_vangogh_weights.hdf5'
st_model = style_transfer(stWeights)

print('Compiling st_model')
st_model.compile(loss='mean_squared_error', optimizer='sgd')

print('Predicting')
# time it
start= time.clock()
for i in range(1):
    results = st_model.predict(X_test)
end= time.clock()
time= (end-start)/1
print("time taken on 1 average call:", time)


print('Dumping results')
column_count = 256
row_count = 256
plane_count = 3
pngWriter = png.Writer(column_count, row_count,
                           greyscale=False,
                           alpha=False,
                           bitdepth=8)

outPath = testPath = dir + '/data/output'
i = 0
for im in results:
    print(im[0])
    fullOutPath = testPath + '/' + str(i) + ".png"
    im = np.clip(im.transpose(1, 2, 0).astype(int), 0, 255)
    f = open(fullOutPath, 'wb')
    pngWriter.write(f, im.reshape((-1, column_count*plane_count)))

    fullOriPath = testPath + '/' + str(i) + "_ori.png"
    im = X_test[i].transpose(1, 2, 0).astype(int)
    f = open(fullOriPath, 'wb')
    pngWriter.write(f, im.reshape((-1, column_count*plane_count)))

    i += 1
