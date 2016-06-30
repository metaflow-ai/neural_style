import os

from keras import backend as K
from keras.callbacks import Callback

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
    def __init__(self, model, chkp_dir, nb_step_chkp=100, max_to_keep=10, keep_checkpoint_every_n_hours=1):
        super(Callback, self).__init__()
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            self.saver = tf.train.Saver(var_list=None, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
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

    def __init__(self, model, log_dir, histogram_freq=0, image_freq=0, audio_freq=0, write_graph=False):
        super(Callback, self).__init__()
        if K._BACKEND != 'tensorflow':
            raise Exception('TensorBoardBatch callback only works '
                            'with the TensorFlow backend.')
        import tensorflow as tf
        self.tf = tf
        import keras.backend.tensorflow_backend as KTF
        self.KTF = KTF

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.image_freq = image_freq
        self.audio_freq = audio_freq
        self.histograms = None
        self.images = None
        self.write_graph = write_graph
        self.iter = 0
        self.scalars = []
        self.images = []
        self.audios = []

        self.model = model
        self.sess = KTF.get_session()

        if self.histogram_freq != 0:
            layers = self.model.layers
            for layer in layers:
                if hasattr(layer, 'name'):
                    layer_name = layer.name
                else:
                    layer_name = layer

                if hasattr(layer, 'W'):
                    name = '{}_W'.format(layer_name)
                    tf.histogram_summary(name, layer.W, collections=['histograms'])
                if hasattr(layer, 'b'):
                    name = '{}_b'.format(layer_name)
                    tf.histogram_summary(name, layer.b, collections=['histograms'])
                if hasattr(layer, 'output'):
                    name = '{}_out'.format(layer_name)
                    tf.histogram_summary(name, layer.output, collections=['histograms'])
        
        if self.image_freq != 0:
            tf.image_summary('input', self.model.input, max_images=2, collections=['images'])
            tf.image_summary('output', self.model.output, max_images=2, collections=['images'])

        if self.audio_freq != 0:
            tf.audio_summary('input', self.model.input, max_outputs=1, collections=['audios'])
            tf.audio_summary('output', self.model.output, max_outputs=1, collections=['audios'])

        if self.write_graph:
            if self.tf.__version__ >= '0.8.0':
                self.writer = self.tf.train.SummaryWriter(self.log_dir, self.sess.graph)
            else:
                self.writer = self.tf.train.SummaryWriter(self.log_dir, self.sess.graph_def)
        else:
            self.writer = self.tf.train.SummaryWriter(self.log_dir)

    def _set_model(self, model):
        return

    def on_train_begin(self, logs):
        self.histograms = self.tf.merge_all_summaries('histograms')
        self.images = self.tf.merge_all_summaries('images')
        self.audios = self.tf.merge_all_summaries('audios')

    def on_batch_end(self, batch, logs):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary(value=[
                self.tf.Summary.Value(
                    tag=name, 
                    simple_value=float(value)
                )]
            )
            self.writer.add_summary(summary, self.iter)

        for name, value in self.scalars:
            summary = self.tf.Summary(value=[
                self.tf.Summary.Value(
                    tag=name, 
                    simple_value=float(K.get_value(value))
                )]
            )
            self.writer.add_summary(summary, self.iter)

        if hasattr(self.model, 'validation_data') and self.model.validation_data:
            if self.model.uses_learning_phase:
                cut_v_data = len(self.model.inputs)
                val_data = self.model.validation_data[:cut_v_data] + [0]
                tensors = self.model.inputs + [K.learning_phase()]
            else:
                val_data = self.model.validation_data
                tensors = self.model.inputs
            self.feed_dict = dict(zip(tensors, val_data))

            if self.image_freq > 0 and batch % self.image_freq == 0:
                result = self.sess.run([self.images], feed_dict=self.feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, self.iter)

            if self.audio_freq > 0 and batch % self.audio_freq == 0:
                result = self.sess.run([self.audios], feed_dict=self.feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, self.iter)

            if self.histogram_freq > 0 and batch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                result = self.sess.run([self.histograms], feed_dict=self.feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, self.iter)

        self.writer.flush()
        self.iter += 1
