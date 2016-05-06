import numpy as np
import os, time
import png

from utils.imutils import load_images
from models.style_tranfer import style_tranfer

dir = os.path.dirname(os.path.realpath(__file__))

print('Loading test set')
X_test = load_images(dir + '/data', 4)
print(X_test.shape)

print('Loading style_tranfer')
stWeights = dir + '/models/st_vangogh_weights.hdf5'
st_model = style_tranfer(stWeights)

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
    fullOutPath = testPath + '/' + str(i) + ".png"
    im = np.clip(im.transpose(1, 2, 0).astype(int), 0, 255)
    f = open(fullOutPath, 'wb')
    pngWriter.write(f, im.reshape((-1, column_count*plane_count)))

    fullOriPath = testPath + '/' + str(i) + "_ori.png"
    im = X_test[i].transpose(1, 2, 0).astype(int)
    f = open(fullOriPath, 'wb')
    pngWriter.write(f, im.reshape((-1, column_count*plane_count)))

    i += 1
