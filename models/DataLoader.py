import numpy as np
import h5py
from keras.utils import np_utils
from random import shuffle


class DataLoader:
    """
    because now the data is too large to be fit into memory, we need to train them in batch
    """
    def __init__(self, batch_size=128, filename="/home/stevenwudi/Documents/Python_Project/OBT/Kernelized_Correlation_Filter/data/OBT100_new_multi_cnn%d.hdf5"):

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

        self.shuffle_train()
        self.shuffle_valid()

    def shuffle_train(self):
        self.pos_train = list(np.random.permutation(self.n_iter_train)*self.batch_size)

    def shuffle_valid(self):
        self.pos_valid = list(np.random.permutation(self.n_iter_valid)*self.batch_size)

    def generate(self, train=True):
        while True:
            if train:
                if len(self.pos_train) == 0:
                    self.shuffle_train()
                pos = self.pos_train.pop()
                x = (self.file["x_train"][pos:pos + self.batch_size]).astype('float32') / 255. - 0.5
                y = (self.file["y_train"][pos:pos + self.batch_size][:, :2]).astype('float32')
                yield (x, y)
            else:
                if len(self.pos_valid) == 0:
                    self.shuffle_valid()
                pos = self.pos_valid.pop() + self.train_num
                x = (self.file["x_train"][pos:pos + self.batch_size]).astype('float32') / 255. - 0.5
                y = (self.file["y_train"][pos:pos + self.batch_size][:, :2]).astype('float32')
                yield (x, y)


class Generator(object):
    def __init__(self, filename,
                 batch_size=128,
                 response_map_shape=[(240, 160), (120, 80), (60, 40), (30, 20), (15, 10)],
                 ):
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