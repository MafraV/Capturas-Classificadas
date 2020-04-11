from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


import numpy as np


import os
import os.path


class SequenceImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png', normalize_seq=True,
                 follow_links=False, interpolation='nearest'):

        self.normalize_seq = normalize_seq
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.classes = classes
        self.class_mode = class_mode
        self.interpolation = interpolation

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)

        # contar os diretorios de classes e armazenar nome dos diretorios
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)

        self.samples = 0
        self.max_seq_length = -1
        # salvar sequencias por classes
        self.sequences_per_class = []
        for class_name in self.class_indices:
            # salvando a quantidade de sequencias por classe = total samples
            cur_cls_dir = os.path.join(directory, class_name)

            seqdirname = sorted(os.listdir(cur_cls_dir))
            self.samples += len(seqdirname)

            self.sequences_per_class.extend([(os.path.join(cur_cls_dir, l),
                                              self.class_indices[class_name])
                                             for l in os.listdir(cur_cls_dir)])
            for seqName in seqdirname:
                curlen = len(os.listdir(os.path.join(cur_cls_dir, seqName)))
                if self.max_seq_length < curlen:
                    self.max_seq_length = curlen

        print("found {} sequences belonging to {} classes".format(
            self.samples, len(self.class_indices)))

        self.classes = np.zeros((self.samples,), dtype='int32')
        sum_n = 0
        for k, v in self.class_indices.items():
            seqdirname = sorted(os.listdir(os.path.join(directory, k)))
            self.classes[sum_n:sum_n + len(seqdirname)] = np.full(len(seqdirname), v)
            sum_n += len(seqdirname)

        super(SequenceImageIterator, self).__init__(self.samples, batch_size,
                                                    shuffle, seed)

    def set_max_length(self, val):
        self.max_seq_length = val

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(self.max_seq_length * len(index_array) *
                           self.target_size[0] * self.target_size[1],
                           dtype=K.floatx())

        batch_x = batch_x.reshape((len(index_array), self.max_seq_length,
                                   self.target_size[0], self.target_size[1], 1))

        grayscale = self.color_mode == "grayscale"

        img = np.zeros(self.target_size[0] * self.target_size[1]) \
            .reshape((self.target_size[0], self.target_size[1]))

        for i1, j in enumerate(index_array):

            seqdir = self.sequences_per_class[j][0]

            last = -1
            seqimgs = sorted(os.listdir(seqdir), key=len)
            curseqlen = self.max_seq_length if self.normalize_seq else len(seqimgs)

            for i2 in range(0, curseqlen):
                i2norm = int(i2 / (self.max_seq_length / len(seqimgs))) if self.normalize_seq else i2
                name = os.path.join(seqdir, seqimgs[i2norm])
                if i2norm != last:
                    img = load_img(name, grayscale=grayscale,
                                   target_size=self.target_size,
                                   interpolation=self.interpolation)
                last = i2norm
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i1][i2] = x

        batch_y = np.zeros((len(index_array), len(self.class_indices)),
                           dtype=K.floatx())
        for it, label in enumerate(index_array):
            cls = self.sequences_per_class[label][1]
            batch_y[it, cls] = 1

        return batch_x, batch_y
    
i_fold = 1
dirname     = 'C:/videos/Train/'
testDirName = 'C:/videos/Test/'
    
seqIt = SequenceImageIterator(dirname, ImageDataGenerator(rescale=1./255),
                              target_size=(50, 50), color_mode='grayscale',
                              batch_size=16, class_mode='categorical',
                              normalize_seq=False)

testSeqIt = SequenceImageIterator(testDirName,
                                  ImageDataGenerator(rescale=1./255),
                                  target_size=(50, 50), color_mode='grayscale',
                                  batch_size=16, class_mode='categorical',
                                  normalize_seq=False, shuffle=False)

max_seq_len = max(seqIt.max_seq_length, testSeqIt.max_seq_length)
testSeqIt.set_max_length(max_seq_len)
seqIt.set_max_length(max_seq_len)


class MaxWeights(Callback):
    def __init__(self):
        self.max_acc = -1
        self.acc_hist = []
        self.weights = []
        
    def on_epoch_end(self, epoch, logs=None):
        curr_acc = logs['val_categorical_accuracy']
        if curr_acc > self.max_acc:
            self.max_acc = curr_acc
            self.weights = self.model.get_weights()
        self.acc_hist.append(curr_acc)
    
# %% classifier
def train_lstm(lstm_units, amount_filters, filters):
    classifier = Sequential()

    classifier.add(TimeDistributed(Conv2D(amount_filters, filters[0],
                                          input_shape=(50, 50, 1),
                                          activation='relu'),
                                   input_shape=(max_seq_len, 50, 50, 1)))
    classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    if len(filters) > 1:
        classifier.add(TimeDistributed(Conv2D(amount_filters, filters[1],
                                              activation='relu')))
        classifier.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    classifier.add(TimeDistributed(Flatten()))

    classifier.add(LSTM(units=lstm_units[0], activation='tanh',
                        return_sequences=True if len(lstm_units) > 1 else False,
                        input_shape=(max_seq_len, 25 * 25)))
    if len(lstm_units) > 1:
        classifier.add(LSTM(units=lstm_units[1], activation='tanh',
                            return_sequences=True if len(lstm_units) > 2
                                             else False))
    if len(lstm_units) > 2:
        classifier.add(LSTM(units=lstm_units[2], activation='tanh'))

    classifier.add(Dense(units=2, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy', metrics.categorical_accuracy])

    classifier.summary()

    max_weights = MaxWeights()
    # Fit the classifier
    classifier.fit_generator(seqIt
                             , callbacks=[max_weights]
                             , steps_per_epoch=150
                             , epochs=15
                             , validation_data=testSeqIt
                             , validation_steps=testSeqIt.samples/16)

    classifier.set_weights(max_weights.weights)
    classifier.save('model_lstm{}_cnn{}.h5'.format(lstm_units, filters))
    classifier.save_weights('model_weights_lstm{}_cnn{}.h5'.format(lstm_units,
                                                                   filters))
    conf_mat = np.zeros((10, 10))
    for idx in range(900):
        spl = testSeqIt._get_batches_of_transformed_samples([idx])
        pred = classifier.predict(spl[0])
        cls = np.argmax(spl[1])
        classifier.reset_states()
        k = np.argmax(pred)
        conf_mat[cls, k] += 1

    out_str = 'lstm units {}\n' + \
              'best val categorical accuracy {}\n' + \
              'confusion mat for best epoch \n{}\n' + \
              'all accuracy per epoch \n{}\n\n'

    print(out_str.format(lstm_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist))
    print(out_str.format(lstm_units, max_weights.max_acc, conf_mat,
                         max_weights.acc_hist), file=open('lstm64.txt', 'w'))

train_lstm([100, 100, 100], 64, [[3, 3], [3, 3]])
K.clear_session()