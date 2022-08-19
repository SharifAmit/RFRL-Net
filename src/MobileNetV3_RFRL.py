from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape

from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Multiply, Add
from keras.utils.vis_utils import plot_model

from keras.layers import Layer, InputSpec


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        original_shape = K.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])

        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))

        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

        X = tf.image.resize_bilinear(X, new_shape)

        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))

        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        """Init.
            size: factor to original shape (ie. original-> size * original).
            target size: target size (ie. original->target).
        """
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)

        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'

        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]

        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class LiteRASSP:
    def __init__(self, input_shape, n_class=19, alpha=1.0, weights=None, backbone='small'):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor (should be 1024 × 2048 or 512 × 1024 according 
                to the paper).
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier for mobilenetV3.
            weights: String, weights for mobilenetv3.
            backbone: String, name of backbone (must be small or large).
        """
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights
        self.backbone = backbone

    def _extract_backbone(self):
        """extract feature map from backbone.
        """
        if self.backbone == 'large':
            from model.mobilenet_v3_large import MobileNetV3_Large

            model = MobileNetV3_Large(self.shape, self.n_class, alpha=self.alpha, include_top=False).build()
            layer_name8 = 'batch_normalization_13'
            layer_name16 = 'add_5'
        elif self.backbone == 'small':
            from model.mobilenet_v3_small import MobileNetV3_Small

            model = MobileNetV3_Small(self.shape, self.n_class, alpha=self.alpha, include_top=False).build()
            layer_name8 = 'batch_normalization_7'
            layer_name16 = 'add_2'
        else:
            raise Exception('Invalid backbone: {}'.format(self.backbone))

        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

        inputs= model.input
        # 1/8 feature map.
        out_feature8 = model.get_layer(layer_name8).output
        # 1/16 feature map.
        out_feature16 = model.get_layer(layer_name16).output

        return inputs, out_feature8, out_feature16

    def build(self, plot=False):
        """build Lite R-ASPP.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        inputs, out_feature8, out_feature16 = self._extract_backbone()

        # branch1
        x1 = Conv2D(128, (1, 1))(out_feature16)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        # branch2
        s = x1.shape

        x2 = AveragePooling2D(pool_size=(49, 49), strides=(16, 20))(out_feature16)
        x2 = Conv2D(128, (1, 1))(x2)
        x2 = Activation('sigmoid')(x2)
        x2 = BilinearUpSampling2D(target_size=(int(s[1]), int(s[2])))(x2)

        # branch3
        x3 = Conv2D(self.n_class, (1, 1))(out_feature8)

        # merge1
        x = Multiply()([x1, x2])
        x = BilinearUpSampling2D(size=(2, 2))(x)
        x = Conv2D(self.n_class, (1, 1))(x)

        # merge2
        x = Add()([x, x3])

        # out
        x = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        if plot:
            plot_model(model, to_file='images/LR_ASPP.png', show_shapes=True)

        return model

class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init
        
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.
        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.
        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

class MobileNetV3_Large(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Large.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        f1, f2 = [], []
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x1 = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x1, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x2 = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x2, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x3 = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x3, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x4 = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x4, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')

        X_Conv = Conv2D(3,(3,3,),strides=(1,1),padding='same', name='upsampled_conv',kernel_initializer=glorot_uniform(seed=0))(x)
        X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same', name = 'autoencoder/Upsample1')(X_Conv)
        X_feat1 = x4
        X_feat1 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat1_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat1)
        f1.append(X_feat1)
        f2.append(X_up)
        X_up = Add()([X_feat1,X_up]) 
        X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample2')(X_up)
        X_feat2 = x3
        X_feat2 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat2_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat2)
        f1.append(X_feat2)
        f2.append(X_up)
        X_up = Add()([X_feat2,X_up])    
        X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample3')(X_up)
        X_feat3 = x2
        X_feat3 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat3_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat3)
        f1.append(X_feat3)
        f2.append(X_up)
        X_up = Add()([X_feat3,X_up])
        X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample4')(X_up)
        X_feat4 = x1
        X_feat4 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat4_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat4)
        f1.append(X_feat4)
        f2.append(X_up)
        X_up = Add()([X_feat4,X_up])
        X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder')(X_up)

        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')


        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,),name='classifier')(x)


        model = Model(inputs, [x,X_up,x])

        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        model.compile(Adam(lr=.0001), loss=['categorical_crossentropy','mean_squared_error',
                                    feature_matching_loss(f_1=f1,f_2=f2)], metrics={'classifier': 'accuracy', 'autoencoder': 'mse'})
        model.summary()
        return model
