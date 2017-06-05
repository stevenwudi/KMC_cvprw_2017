"""
This is a python reimplementation of the paper
@inproceedings{wu2017cvprw,
  title={Kernalised Multi-resolution Convnet for Visual Tracking},
  author={Wu, Di and Wenbin, Zou and Xia, Li and Yong, Zhao},
  booktitle={Proc. Conference on Computer Vision and Pattern Recognition (CVPR) Workshop},
  year={2017}
}
author: DI WU
email: stevenwudi@gmail.com
2017/06/05
"""
import numpy as np
from scipy.misc import imresize
import keras


class KMCTracker:
    def __init__(self, feature_type='multi_cnn',
                 model_proto=None,
                 model_path='./trained_models/CNN_Model_OBT100_multi_cnn_final.h5',
                 feature_bandwidth_sigma=0.2,
                 spatial_bandwidth_sigma_factor=float(1/16.),
                 adaptation_rate_scale_range_max=0.025,
                 adaptation_rate_range_max=0.0025,
                 sub_feature_type="",
                 sub_sub_feature_type="",
                 padding=2.2,
                 lambda_value=1e-4):
        """
        object_example is an image showing the object to track
        feature_type:
            "raw pixels":
            "hog":
            "CNN":
        """
        # parameters according to the paper --
        self.padding = padding  # extra area surrounding the target
        self.lambda_value = lambda_value  # regularization
        self.spatial_bandwidth_sigma_factor = spatial_bandwidth_sigma_factor
        self.feature_type = feature_type
        self.patch_size = []
        self.output_sigma = []
        self.cos_window = []
        self.pos = []
        self.x = []
        self.alphaf = []
        self.xf = []
        self.yf = []
        self.im_crop = []
        self.response = []
        self.target_out = []
        self.target_sz = []
        self.vert_delta = 0
        self.horiz_delta = 0
        # OBT dataset need extra definition
        self.name = 'KCF_' + feature_type
        self.fps = -1
        self.res = []
        self.im_sz = []
        self.first_patch_sz = []
        self.first_target_sz = []
        self.currentScaleFactor = 1
        self.model_proto = model_proto
        self.adaptation_rate_scale_range_max = adaptation_rate_scale_range_max
        self.sub_feature_type = sub_feature_type
        self.sub_sub_feature_type = sub_sub_feature_type

        # following is set according to Table 2:
        if self.feature_type == 'multi_cnn':
            import keras
            from keras import backend as K
            from keras.applications.vgg19 import VGG19
            self.base_model = VGG19(include_top=False, weights='imagenet')
            self.extract_model_function = K.function([self.base_model.input],
                                                         [self.base_model.get_layer('block1_conv2').output,
                                                           self.base_model.get_layer('block2_conv2').output,
                                                           self.base_model.get_layer('block3_conv4').output,
                                                           self.base_model.get_layer('block4_conv4').output,
                                                           self.base_model.get_layer('block5_conv4').output
                                                          ])

            # we first resize all the response maps to a size of 40*60 (store the resize scale)
            # because average target size is 81 *52
            self.resize_size = (240, 160)
            self.cell_size = 4
            self.response_size = [self.resize_size[0] / self.cell_size,
                                  self.resize_size[1] / self.cell_size]
            self.feature_bandwidth_sigma = feature_bandwidth_sigma
            self.adaptation_rate = adaptation_rate_range_max
            self.stability = np.ones(5)
            # store pre-computed cosine window, here is a multiscale CNN, here we have 5 layers cnn:
            self.cos_window = []
            self.y = []
            self.yf = []
            self.response_all = []
            self.max_list = []
            self.feature_bandwidth_sigma = feature_bandwidth_sigma

            for i in range(5):
                cos_wind_sz = np.divide(self.resize_size, 2**i)
                self.cos_window.append(np.outer(np.hanning(cos_wind_sz[0]), np.hanning(cos_wind_sz[1])))
                grid_y = np.arange(cos_wind_sz[0]) - np.floor(cos_wind_sz[0] / 2)
                grid_x = np.arange(cos_wind_sz[1]) - np.floor(cos_wind_sz[1] / 2)
                # desired output (gaussian shaped), bandwidth proportional to target size
                output_sigma = np.sqrt(np.prod(cos_wind_sz)) * self.spatial_bandwidth_sigma_factor
                rs, cs = np.meshgrid(grid_x, grid_y)
                y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
                self.y.append(y)
                self.yf.append(self.fft2(y))

            if self.model_proto:
                    import models.CNN
                    self.multi_cnn_model = getattr(models.CNN, self.model_proto)()
                    self.multi_cnn_model.load_weights(model_path)
        if self.sub_feature_type == 'dsst':
            # this method adopts from the paper  Martin Danelljan, Gustav Hger, Fahad Shahbaz Khan and Michael Felsberg.
            # "Accurate Scale Estimation for Robust Visual Tracking". (BMVC), 2014.
            # The project website is: http: // www.cvl.isy.liu.se / research / objrec / visualtracking / index.html
            self.scale_step = 1.01
            self.nScales = 33
            self.scaleFactors = self.scale_step ** (np.ceil(self.nScales * 1.0 / 2) - range(1, self.nScales + 1))
            self.scale_window = np.hanning(self.nScales*6+1)[self.nScales+33/2+2:self.nScales*5-33/2:3]
            self.scale_sigma_factor = 1. / 4
            self.scale_sigma = self.nScales / np.sqrt(self.nScales) * self.scale_sigma_factor
            self.ys = np.exp(
                -0.5 * ((range(1, self.nScales + 1) - np.ceil(self.nScales * 1.0 / 2)) ** 2) / self.scale_sigma ** 2)
            self.ysf = np.fft.fft(self.ys)
            self.min_scale_factor = []
            self.max_scale_factor = []
            self.xs = []
            self.xsf = []
            self.sf_num = []
            self.sf_den = []
            # we use linear kernel as in the BMVC2014 paper
            self.new_sf_num = []
            self.new_sf_den = []
            self.scale_response = []
            self.lambda_scale = 1e-2
            self.adaptation_rate_scale = adaptation_rate_scale_range_max

        self.name = "KMC_" + self.feature_type
        self.name = "KMC_" + self.sub_feature_type

    def train(self, im, init_rect):
        """
        :param im: image should be of 3 dimension: M*N*C
        :param pos: the centre position of the target
        :param target_sz: target size
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res.append(init_rect)
        # for scaling, we always need to set it to 1
        self.currentScaleFactor = 1
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.first_target_sz = self.target_sz  # because we might introduce the scale changes in the detection
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.first_patch_sz = np.array(self.patch_size).astype(int)   # because we might introduce the scale changes in the detection
        # desired output (gaussian shaped), bandwidth proportional to target size
        self.im_sz = im.shape[:2]
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.x = self.get_features()
        self.xf = self.fft2(self.x)
        self.alphaf = []
        for i in range(len(self.x)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i])
            self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + self.lambda_value))

        if self.sub_feature_type == 'dsst':
            self.min_scale_factor = self.scale_step ** (
            np.ceil(np.log(max(5. / self.patch_size)) / np.log(self.scale_step)))
            self.max_scale_factor = self.scale_step ** (
            np.log(min(np.array(self.im_sz[:2]).astype(float) / self.target_sz)) / np.log(self.scale_step))
            self.xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
            self.xsf = np.fft.fftn(self.xs, axes=[0])
            # we use linear kernel as in the BMVC2014 paper
            self.sf_num = np.multiply(self.ysf[:, None], np.conj(self.xsf))
            self.sf_den = np.real(np.sum(np.multiply(self.xsf, np.conj(self.xsf)), axis=1))

    def detect(self, im):
        """
        Note: we assume the target does not change in scale, hence there is no target size
        :param im: image should be of 3 dimension: M*N*C
        :return:
        """
        # Quote from BMVC2014paper: Danelljan:
        # "In visual tracking scenarios, the scale difference between two frames is typically smaller compared to the
        # translation. Therefore, we first apply the translation filter hf given a new frame, afterwards the scale
        # filter hs is applied at the new target location.

        # extract and pre-process subwindow
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)
        self.response = []
        for i in range(len(z)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i], zf[i], z[i])
            kf = self.fft2(k)
            self.response.append(np.real(np.fft.ifft2(np.multiply(self.alphaf[i], kf))))

        response_all = np.zeros(shape=(5, self.resize_size[0], self.resize_size[1]))
        self.max_list = [np.max(x) for x in self.response]
        for i in range(len(self.response)):
            response_all[i, :, :] = imresize(self.response[i], size=self.resize_size)
            response_all[i, :, :] *= self.max_list[i]
        self.response_all = response_all.astype('float32') / 255.

        # prediction
        inputs = []
        for x in self.response:
            inputs.append(np.expand_dims(np.expand_dims(np.array(x).astype('float32'), 0), 3))
        pos_move = self.multi_cnn_model.predict(inputs)

        self.vert_delta, self.horiz_delta = [self.target_sz[0] * pos_move[0][0], self.target_sz[1] * pos_move[0][1]]
        self.pos = [self.pos[0] + self.target_sz[0] * pos_move[0][0],
                    self.pos[1] + self.target_sz[1] * pos_move[0][1]]
        self.pos = [max(self.target_sz[0] / 2, min(self.pos[0], self.im_sz[0] - self.target_sz[0] / 2)),
                    max(self.target_sz[1] / 2, min(self.pos[1], self.im_sz[1] - self.target_sz[1] / 2))]

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        # we update the scale from here
        if self.sub_feature_type == 'dsst':
            xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
            xsf = np.fft.fftn(xs, axes=[0])
            # calculate the correlation response of the scale filter
            scale_response_fft = np.divide(np.multiply(self.sf_num, xsf),
                                           (self.sf_den[:, None] + self.lambda_scale))
            scale_reponse = np.real(np.fft.ifftn(np.sum(scale_response_fft, axis=1)))
            recovered_scale = np.argmax(scale_reponse)
            # update the scale
            self.currentScaleFactor *= self.scaleFactors[recovered_scale]
            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            elif self.currentScaleFactor > self.max_scale_factor:
                self.currentScaleFactor = self.max_scale_factor
            # we only update the target size here.
            new_target_sz = np.multiply(self.currentScaleFactor, self.first_target_sz)
            self.pos -= (new_target_sz-self.target_sz)/2
            self.target_sz = new_target_sz
            self.patch_size = np.multiply(self.target_sz, (1 + self.padding))

        ###############################
        # we update the model from here
        ###############################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)
        for i in range(len(x_new)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
            kf = self.fft2(k)
            alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
            self.x[i] = (1 - self.adaptation_rate) * self.x[i] + self.adaptation_rate * x_new[i]
            self.xf[i] = (1 - self.adaptation_rate) * self.xf[i] + self.adaptation_rate * xf_new[i]
            self.alphaf[i] = (1 - self.adaptation_rate) * self.alphaf[i] + self.adaptation_rate * alphaf_new

        if self.sub_feature_type == 'dsst':
            xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
            xsf = np.fft.fftn(xs, axes=[0])
            # we use linear kernel as in the BMVC2014 paper
            new_sf_num = np.multiply(self.ysf[:, None], np.conj(xsf))
            new_sf_den = np.real(np.sum(np.multiply(xsf, np.conj(xsf)), axis=1))
            self.sf_num = (1 - self.adaptation_rate_scale) * self.sf_num + self.adaptation_rate_scale * new_sf_num
            self.sf_den = (1 - self.adaptation_rate_scale) * self.sf_den + self.adaptation_rate_scale * new_sf_den

        # we also require the bounding box to be within the image boundary
        self.res.append([min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
                         self.target_sz[1], self.target_sz[0]])

        return self.pos

    def dense_gauss_kernel(self, sigma, xf, x, zf=None, z=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        :param sigma: feature bandwidth sigma
        :param x:
        :param y: if y is None, then we calculate the auto-correlation
        :return:
        """
        N = xf.shape[0]*xf.shape[1]
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if zf is None:
            # auto-correlation of x
            zf = xf
            zz = xx
        else:
            zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

        xyf = np.multiply(zf, np.conj(xf))
        if self.feature_type == 'raw' or self.feature_type == 'dsst':
            if len(xyf.shape) == 3:
                xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
            elif len(xyf.shape) == 2:
                xyf_ifft = np.fft.ifft2(xyf)
            # elif len(xyf.shape) == 4:
            #     xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=3))
        elif self.feature_type == 'hog':
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50' \
                or self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn' \
                or self.feature_type =='multi_cnn' or self.feature_type =='HDT':
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))

        row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
        xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        c = np.real(xy_complex)
        d = np.real(xx) + np.real(zz) - 2 * c
        k = np.exp(-1. / sigma**2 * np.maximum(0, d) / N)

        return k

    def get_subwindow(self, im, pos, sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        # extract image

        if self.feature_type == 'raw' or self.feature_type == 'dsst':
            out = im[np.ix_(ys, xs)]
            # introduce scaling, here, we need them to be the same size
            if np.all(self.first_patch_sz == out.shape[:2]):
                return out
            else:
                out = imresize(out, self.first_patch_sz)
                return out / 255.
        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50' or \
             self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn' \
                or self.feature_type == 'multi_cnn' or self.feature_type == 'HDT':
            c = np.array(range(3))
            out = im[np.ix_(ys, xs, c)]
            # if self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn':
            #     from keras.applications.vgg19 import preprocess_input
            #     x = imresize(out.copy(), self.resize_size)
            #     out = np.multiply(x, self.cos_window_patch[:, :, None])
            return out

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        if type(x) == list:
            x = [np.fft.fft2(f, axes=(0,1)) for f in x]
            return x
        else:
            return np.fft.fft2(x, axes=(0, 1))

    def get_features(self):
        """
        :param im: input image
        :return:
        """
        if self.feature_type == 'raw':
            #using only grayscale:
            if len(self.im_crop.shape) == 3:
                if self.sub_feature_type == 'gray':
                    img_gray = np.mean(self.im_crop, axis=2)
                    img_gray = img_gray - img_gray.mean()
                    features = np.multiply(img_gray, self.cos_window)
                else:
                    img_colour = self.im_crop - self.im_crop.mean()
                    features = np.multiply(img_colour, self.cos_window[:, :, None])

        elif self.feature_type == 'dsst':
            img_colour = self.im_crop - self.im_crop.mean()
            features = np.multiply(img_colour, self.cos_window[:, :, None])

        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50':
            if self.feature_type == 'vgg':
                from keras.applications.vgg19 import preprocess_input
            elif self.feature_type == 'resnet50':
                from keras.applications.resnet50 import preprocess_input
            x = np.expand_dims(self.im_crop.copy(), axis=0)
            x = preprocess_input(x)
            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
            features = np.multiply(features, self.cos_window[:, :, None])

        elif self.feature_type == 'vgg_rnn' or self.feature_type=='cnn':
            from keras.applications.vgg19 import preprocess_input
            x = imresize(self.im_crop.copy(), self.resize_size)
            x = x.transpose((2, 0, 1)).astype(np.float64)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
            features = np.multiply(features, self.cos_window[:, :, None])

        elif self.feature_type == "multi_cnn":
            from keras.applications.vgg19 import preprocess_input
            x = imresize(self.im_crop.copy(), self.resize_size)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            x = preprocess_input(x)
            if keras.backend._backend == 'theano':
                features_list = self.extract_model_function(x)
            else:
                features_list = self.extract_model_function([x])
            for i, features in enumerate(features_list):
                features = np.squeeze(features)
                features = (features - features.min()) / (features.max() - features.min())
                features_list[i] = np.multiply(features, self.cos_window[i][:, :, None])
            return features_list
        elif self.feature_type == "HDT":
            from keras.applications.vgg19 import preprocess_input
            x = imresize(self.im_crop.copy(), self.resize_size)
            x = x.transpose((2, 0, 1)).astype(np.float64)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features_list = self.extract_model_function(x)
            for i, features in enumerate(features_list):
                features = np.squeeze(features)
                features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
                features_list[i] = np.multiply(features, self.cos_window[i][:, :, None])
                #features_list[i] = np.multiply(features.transpose(1, 2, 0), self.cos_window[i][:, :, None])
            return features_list
        else:
            assert 'Non implemented!'

        if not (self.sub_feature_type=="" or self.feature_correlation is None):
            features = np.multiply(features, self.feature_correlation[None, None, :])
        return features

    def get_scale_sample(self, im, scaleFactors):
        from pyhog import pyhog
        resized_im_array = []
        for i, s in enumerate(scaleFactors):
            patch_sz = np.floor(self.first_target_sz * s)
            im_patch = self.get_subwindow(im, self.pos, patch_sz)  # extract image
            # because the hog output is (dim/4)-2>1:
            if self.first_target_sz.min() < 12:
                scale_up_factor = 12. / np.min(self.first_target_sz)
                im_patch_resized = imresize(im_patch, np.asarray(self.first_target_sz * scale_up_factor).astype('int'))
            else:
                im_patch_resized = imresize(im_patch, self.first_target_sz)  #resize image to model size
            features_hog = pyhog.features_pedro(im_patch_resized.astype(np.float64)/255.0, 4)
            resized_im_array.append(np.multiply(features_hog.flatten(), self.scale_window[i]))
        return np.asarray(resized_im_array)

    def train_cnn(self, frame, im, init_rect, img_rgb_next, next_rect, x_train, y_train, count):

        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        # OTB is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.next_target_sz = np.asarray(next_rect[2:])
        self.next_target_sz = self.next_target_sz[::-1]
        self.scale_change = np.divide(np.array(self.next_target_sz).astype(float), self.target_sz)
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.im_sz = im.shape[:2]

        if frame == 0:
            self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
            self.x = self.get_features()
            self.xf = self.fft2(self.x)
            self.alphaf = []
            for i in range(len(self.x)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i])
                self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + self.lambda_value))

        ###################### Next frame #####################################
        #t0 = time.clock()
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)
        #print(time.clock() - t0, "Feature process time")
        self.response = []
        for i in range(len(z)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i], zf[i], z[i])
            kf = self.fft2(k)
            self.response.append(np.real(np.fft.ifft2(np.multiply(self.alphaf[i], kf))))

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        self.pos_next = [next_rect[1] + next_rect[3] / 2., next_rect[0] + next_rect[2] / 2.]
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos_next, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)
        for i in range(len(x_new)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
            kf = self.fft2(k)
            alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
            self.x[i] = (1 - self.adaptation_rate) * self.x[i] + self.adaptation_rate * x_new[i]
            self.xf[i] = (1 - self.adaptation_rate) * self.xf[i] + self.adaptation_rate * xf_new[i]
            self.alphaf[i] = (1 - self.adaptation_rate) * self.alphaf[i] + self.adaptation_rate * alphaf_new

        # we fill the matrix with zeros first
        response_all = np.zeros(shape=(5, self.resize_size[0], self.resize_size[1]))

        for i in range(len(self.response)):
            response_all[i, :self.response[i].shape[0], :self.response[i].shape[1]] = self.response[i]

        x_train[count, :, :, :] = response_all
        self.pos_next = [next_rect[1] + next_rect[3] / 2., next_rect[0] + next_rect[2] / 2.]
        pos_move = np.array([(self.pos_next[0] - self.pos[0]) * 1.0 / self.target_sz[0],
                             (self.pos_next[1] - self.pos[1]) * 1.0 / self.target_sz[1]])
        y_train[count, :] = np.concatenate([pos_move, self.scale_change])
        count += 1
        return x_train, y_train, count

        # ('feature time:', 0.07054710388183594)
        # ('fft2:', 0.22904396057128906)
        # ('guassian kernel + fft2: ', 0.20537400245666504)
