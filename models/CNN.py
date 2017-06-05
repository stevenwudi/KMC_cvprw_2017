from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import concatenate
import tensorflow as tf


def l1_smooth_loss(y_true, y_pred):
        """Compute L1-smooth loss.

        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).

        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

        # References
            https://arxiv.org/abs/1504.08083
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)


def cnn_cifar_batchnormalisation(image_shape):

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2))
    model.name = 'cnn_cifar_batchnormalisation'

    return model


def cnn_hiararchical_batchnormalisation():

    input_1 = Input(shape=(240, 160, 1), name='input_1')
    input_2 = Input(shape=(120, 80, 1), name='input_2')
    input_3 = Input(shape=(60, 40, 1), name='input_3')
    input_4 = Input(shape=(30, 20, 1), name='input_4')
    input_5 = Input(shape=(15, 10, 1), name='input_5')

    x1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    x2 = concatenate([x1, input_2])
    x2 = Conv2D(32, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = AveragePooling2D(pool_size=(2, 2))(x2)

    x3 = concatenate([x2, input_3])
    x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = AveragePooling2D(pool_size=(2, 2))(x3)

    x4 = concatenate([x3, input_4])
    x4 = Conv2D(64, (3, 3), padding='same', activation='relu')(x4)
    x4 = BatchNormalization()(x4)
    x4 = AveragePooling2D(pool_size=(2, 2))(x4)

    x5 = concatenate([x4, input_5])
    x5 = Conv2D(64, (3, 3), padding='same', activation='relu')(x5)
    x5 = BatchNormalization()(x5)
    x5 = AveragePooling2D(pool_size=(2, 2))(x5)

    x6 = Flatten()(x5)
    x6 = Dense(512, activation='relu')(x6)
    x6 = BatchNormalization()(x6)
    out = Dense(2)(x6)

    model = Model(inputs=[input_1, input_2, input_3, input_4, input_5],
                  outputs=[out])

    model.name = 'cnn_hiararchical_batchnormalisation'

    return model


