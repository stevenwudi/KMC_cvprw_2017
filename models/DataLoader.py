import numpy as np
import h5py
from keras.utils import np_utils
from random import shuffle


class DataLoader:
    """
    because now the data is too large to be fit into memory, we need to train them in batch
    """
    def __init__(self, batch_size=128, filename="./data/OBT100_multi_cnn%d.hdf5"):

        self.file = h5py.File(filename, "r", driver="family", memb_size=2 ** 32 - 1)
        self.batch_size = batch_size
        self.total_num = self.file["x_train"].shape[0]
        self.image_shape = self.file["x_train"][0, :].shape
        # using 10% data for validation
        self.train_num = int(self.total_num * 0.9)
        self.valid_num = int(self.total_num * 0.1)
        self.n_iter_train = int(np.floor(self.train_num/float(batch_size)))
        self.n_iter_valid = int(np.floor(self.valid_num/float(batch_size)))
        self.pos_train = []
        self.pos_valid = []

        # we cast the translational to a range of class 12
        translation_range = [-0.5, 0.5]
        translation_buckets = 20
        translation_step = (translation_range[1] - translation_range[0])/translation_buckets

        translation_hist = [translation_range[0]-translation_step/2 + translation_step*x for x in range(translation_buckets+1)]
        # because -5 is the minimum range for translation for our extraction
        translation_hist.insert(0, -5)
        translation_value = [x + translation_step/2 for x in translation_hist]
        translation_value[0] = translation_range[0] - translation_step
        translation_value.append(translation_range[1] + translation_step)
        self.translation_hist = translation_hist
        self.translation_value = translation_value

        # we cast the scale into a range of class of 12 (10 plus top & bottom)
        scale_range = [0.8, 1.2]
        scale_buckets = 10
        scale_step = (scale_range[1] - scale_range[0]) / scale_buckets
        scale_hist = [scale_range[0] - scale_step/2 + scale_step*x for x in range(scale_buckets+1)]
        scale_hist.insert(0, -0.1)
        scale_value = [x+scale_step/2 for x in scale_hist]
        scale_value[0] = scale_range[0] - scale_step
        scale_value.append(scale_range[1] + scale_step)
        self.scale_hist = scale_hist
        self.scale_value = scale_value

        self.shuffle_train()
        self.shuffle_valid()

    def next_train_batch(self):
        if len(self.pos_train) == 0:
            self.shuffle_train()
        pos = self.pos_train.pop()
        x = (self.file["x_train"][pos:pos+self.batch_size]).astype('float32') / 255. - 0.5
        y = (self.file["y_train"][pos:pos+self.batch_size][:, :2]).astype('float32')
        return x, y

    def get_valid(self):
        x = (self.file["x_train"][self.train_num + 1:]).astype('float32') / 255. - 0.5
        y = (self.file["y_train"][self.train_num + 1:][:, :2]).astype('float32')
        return x, y

    def next_train_batch_with_scale_buckets(self):
        if len(self.pos_train) == 0:
            self.shuffle_train()
        pos = self.pos_train.pop()
        x = (self.file["x_train"][pos:pos+self.batch_size]).astype('float32') / 255. - 0.5
        translation_x = (self.file["y_train"][pos:pos+self.batch_size][:, 0]).astype('float32')
        translation_y = (self.file["y_train"][pos:pos+self.batch_size][:, 1]).astype('float32')
        translation_x_class = np.zeros(shape=translation_x.shape)
        translation_y_class = np.zeros(shape=translation_y.shape)
        for i, t in enumerate(translation_x):
            translation_x_class[i] = np.asarray(np.where(t > self.translation_hist)).max()

        for i, t in enumerate(translation_y):
            translation_y_class[i] = np.asarray(np.where(t > self.translation_hist)).max()

        ys = (self.file["y_train"][pos:pos+self.batch_size][:, 2:].sum(axis=1) / 2.).astype('float32')
        y_scale_class = np.zeros(shape=ys.shape)
        for i, y_temp in enumerate(ys):
            y_scale_class[i] = np.asarray(np.where(y_temp > self.scale_hist)).max()

        translation_x_class = np_utils.to_categorical(translation_x_class, len(self.translation_value))
        translation_y_class = np_utils.to_categorical(translation_y_class, len(self.translation_value))

        y_scale_class = np_utils.to_categorical(y_scale_class, len(self.scale_value))

        return x, translation_x_class, translation_y_class, y_scale_class

    def get_valid_class(self):
        x = (self.file["x_train"][self.train_num + 1:]).astype('float32') / 255. - 0.5
        translation_x = (self.file["y_train"][self.train_num + 1:][:, 0]).astype('float32')
        translation_y = (self.file["y_train"][self.train_num + 1:][:, 1]).astype('float32')
        translation_x_class = np.zeros(shape=translation_x.shape)
        translation_y_class = np.zeros(shape=translation_y.shape)
        for i, t in enumerate(translation_x):
            translation_x_class[i] = np.asarray(np.where(t > self.translation_hist)).max()

        for i, t in enumerate(translation_y):
            translation_y_class[i] = np.asarray(np.where(t > self.translation_hist)).max()

        ys = (self.file["y_train"][self.train_num + 1:][:, 2:].sum(axis=1) / 2.).astype('float32')
        y_scale_class = np.zeros(shape=ys.shape)
        for i, y_temp in enumerate(ys):
            y_scale_class[i] = np.asarray(np.where(y_temp > self.scale_hist)).max()

        translation_x_class = np_utils.to_categorical(translation_x_class, len(self.translation_value))
        translation_y_class = np_utils.to_categorical(translation_y_class, len(self.translation_value))

        y_scale_class = np_utils.to_categorical(y_scale_class, len(self.scale_value))

        return x, translation_x_class, translation_y_class, y_scale_class

    def shuffle_train(self):
        self.pos_train = list(np.random.permutation(self.n_iter_train)*self.batch_size)

    def shuffle_valid(self):
        self.pos_valid = list(np.random.permutation(self.n_iter_valid)*self.batch_size)


class Generator(object):
    def __init__(self, batch_size=128,
                 filename="./data/OBT100_multi_cnn%d.hdf5",
                 response_map_shape=[(240, 160), (120, 80), (60, 40), (30, 20), (15, 10)]):
        self.file = h5py.File(filename, "r", driver="family", memb_size=2 ** 32 - 1)
        self.batch_size = batch_size
        self.total_num = self.file["x_train"].shape[0]
        # using 10% data for validation
        self.train_num = int(self.total_num * 0.9)
        self.valid_num = int(self.total_num * 0.1)
        self.train_keys = range(self.train_num)
        self.val_keys = range(self.train_num, self.train_num+self.valid_num)
        self.train_batches = len(self.train_keys) / self.batch_size
        self.val_batches = len(self.val_keys) / self.batch_size
        self.response_map_shape = response_map_shape

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys

            count = 0
            targets = []
            input_dict = {}
            for layer in range(len(self.response_map_shape)):
                input_dict[layer] = np.zeros(shape=(self.batch_size,
                                                    self.response_map_shape[layer][0],
                                                    self.response_map_shape[layer][1],
                                                    1))
            for key in keys:
                img_all = self.file["x_train"][key].astype('float32')
                y = self.file["y_train"][key, :2].astype('float32')
                for layer in range(len(self.response_map_shape)):
                    input_dict[layer][count, :, :, 0] = img_all[layer,
                                               :self.response_map_shape[layer][0],
                                               :self.response_map_shape[layer][1]]

                targets.append(y)
                count += 1
                if count == self.batch_size:
                    inputs = []
                    for layer in range(len(self.response_map_shape)):
                        inputs.append(np.array(input_dict[layer]))
                    tmp_targets = np.array(targets)
                    count = 0
                    targets = []
                    yield (inputs, tmp_targets)