import os

from keras import backend as K
from keras.callbacks import Callback

from utils.imutils import load_images, save_image_st, save_image
from utils.general import export_model

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

class HistoryBatch(Callback):
    '''Callback that records events
    into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    '''
    def on_train_begin(self, logs={}):
        self.batch = []
        self.history = {}

    def on_batch_end(self, batch, logs={}):
        self.batch.append(batch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

class ModelCheckpointBatch(Callback):
    '''Save the model every nb_step_chkp thx to Tensorflow saver
    '''
    def __init__(self, model, chkp_dir, nb_step_chkp=100):
        super(Callback, self).__init__()
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            self.saver = tf.train.Saver(var_list=None)
        else:
            self.saver = None

        self.model = model
        self.archi = model.to_json()
        self.chkp_dir = chkp_dir
        if not os.path.isdir(self.chkp_dir): 
            os.makedirs(self.chkp_dir)
        self.global_step = 0
        self.nb_step_chkp = nb_step_chkp

    def _set_model(self, model):
        # The model is already set
        return

    def on_train_begin(self, logs={}):
        export_model(self.model, self.chkp_dir, saver=self.saver, global_step=0)

    def on_batch_end(self, batch, logs={}):
        self.global_step += 1
        if self.global_step % self.nb_step_chkp == 0:
            export_model(self.model, self.chkp_dir, saver=self.saver, global_step=self.global_step)

    def on_train_end(self, logs={}):
        # When the freeze script will ne more stable, we can remove the global_step var
        # export_model(self.model, self.chkp_dir, saver=self.saver)
        export_model(self.model, self.chkp_dir, saver=self.saver, global_step=self.global_step)        

class TensorBoardBatch(Callback):
    ''' Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by tensorboard
        histogram_freq: frequency (in batchs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in tensorboard. The log file can
            become quite large when write_graph is set to True.
    '''

    def __init__(self, model, log_dir, histogram_freq=100, write_graph=False):
        super(Callback, self).__init__()
        if K._BACKEND != 'tensorflow':
            raise Exception('TensorBoardBatch callback only works '
                            'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph

        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        self.model = model
        self.sess = KTF.get_session()
        if self.histogram_freq and self.merged is None:
            layers = self.model.layers
            for layer in layers:
                if hasattr(layer, 'W'):
                    tf.histogram_summary('{}_W'.format(layer), layer.W)
                if hasattr(layer, 'b'):
                    tf.histogram_summary('{}_b'.format(layer), layer.b)
                if hasattr(layer, 'output'):
                    tf.histogram_summary('{}_out'.format(layer),
                                         layer.output)
        self.merged = tf.merge_all_summaries()
        if self.write_graph:
            if tf.__version__ >= '0.8.0':
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph_def)
        else:
            self.writer = tf.train.SummaryWriter(self.log_dir)

    def _set_model(self, model):
        return
        
    def on_batch_end(self, batch, logs={}):
        import tensorflow as tf

        if batch % self.histogram_freq == 0:
            if self.model.validation_data and self.histogram_freq:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.model.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, batch)

            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = float(value)
                summary_value.tag = name
                self.writer.add_summary(summary, batch)
            self.writer.flush()