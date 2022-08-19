import tensorflow as tf
from keras.layers import Layer, Reshape, Activation, Conv2D, Conv2DTranspose, SeparableConv2D, Dropout, Dense, UpSampling2D
from keras.layers import Input, Add, Concatenate, Lambda,LeakyReLU,GlobalAveragePooling2D, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import glorot_uniform
from src.feat_rep_sim import *
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

def VGG16_RFRL(input_size,num_of_classes):    
    f1, f2 = [], []
    input_shape = (image_size,image_size,3)
    base_model = VGG16(weights='imagenet',include_top=False,pooling=None,input_shape=input_shape)
    
    X = base_model.output
    
    X_Conv = Conv2D(3,(3,3,),strides=(1,1),padding='same', name='upsampled_conv',kernel_initializer=glorot_uniform(seed=0))(X)
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same', name = 'autoencoder/Upsample1')(X_Conv)
    X_feat1 = base_model.get_layer('block4_pool').output
    X_feat1 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat1_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat1)
    f1.append(X_feat1)
    f2.append(X_up)
    X_up = Add()([X_feat1,X_up]) 
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample2')(X_up)
    X_feat2 = base_model.get_layer('block3_pool').output
    X_feat2 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat2_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat2)
    f1.append(X_feat2)
    f2.append(X_up)
    X_up = Add()([X_feat2,X_up])    
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample3')(X_up)
    X_feat3 = base_model.get_layer('block2_pool').output
    X_feat3 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat3_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat3)
    f1.append(X_feat3)
    f2.append(X_up)
    X_up = Add()([X_feat3,X_up])
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample4')(X_up)
    X_feat4 = base_model.get_layer('block1_pool').output
    X_feat4 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat4_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat4)
    f1.append(X_feat4)
    f2.append(X_up)
    X_up = Add()([X_feat4,X_up])
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder')(X_up)
    X = GlobalAveragePooling2D(name='global_avg_pool')(X)
    X = Dense(256, name='Dense_1')(X)
    X = Dense(num_of_classes, name='Dense_2')(X)
    X = Activation('softmax', name='classifier')(X)
    
    #fm1 = feature_matching_loss(f_1=f1,f_2=f2)
    model = Model(inputs=base_model.input, outputs=[X,X_up,X], name='')

    model.compile(Adam(lr=.0001), loss=['categorical_crossentropy','mean_squared_error',
                                        feature_matching_loss(f_1=f1,f_2=f2)], metrics={'classifier': 'accuracy', 'autoencoder': 'mse'})
    
    model.summary()
    return model

def ResNet50_RFRL(input_size,num_of_classes):    
    
    f1, f2 = [], []
    input_shape = (image_size,image_size,3)
    base_model = ResNet50(weights='imagenet',include_top=False,pooling=None,input_shape=input_shape)
    X = base_model.output
    
    X_Conv = Conv2D(3,(3,3,),strides=(1,1),padding='same', name='upsampled_conv',kernel_initializer=glorot_uniform(seed=0))(X)
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample1')(X_Conv)
    X_feat1 = base_model.get_layer('activation_40').output
    X_feat1 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat1_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat1)
    f1.append(X_feat1)
    f2.append(X_up)
    X_up = Add()([X_feat1,X_up]) 
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample2')(X_up)
    X_feat2 = base_model.get_layer('activation_22').output
    X_feat2 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat2_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat2)
    f1.append(X_feat2)
    f2.append(X_up)
    X_up = Add()([X_feat2,X_up])    
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample3')(X_up)
    X_feat3 = base_model.get_layer('activation_10').output
    X_feat3 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat3_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat3)
    f1.append(X_feat3)
    f2.append(X_up)
    X_up = Add()([X_feat3,X_up])
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample4')(X_up)
    X_feat4 = base_model.get_layer('activation_1').output
    X_feat4 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat4_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat4)
    f1.append(X_feat4)
    f2.append(X_up)
    X_up = Add()([X_feat4,X_up])
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder')(X_up)
    X = GlobalAveragePooling2D(name='global_avg_pool')(X)
    X = Dense(256, name='Dense_1')(X)
    X = Dense(num_of_classes, name='Dense_2')(X)
    X = Activation('softmax', name='classifier')(X)
    
    #fm1 = feature_matching_loss(f_1=f1,f_2=f2)
    model = Model(inputs=base_model.input, outputs=[X,X_up,X], name='')

    model.compile(Adam(lr=.0001), loss=['categorical_crossentropy','mean_squared_error',
                                        feature_matching_loss(f_1=f1,f_2=f2)], metrics={'classifier': 'accuracy', 'autoencoder': 'mse'})
    
    model.summary()
    
    return model

def MobileNetV2_RFRL(input_size,num_of_classes):    
    f1, f2 = [], []
    input_shape = (image_size,image_size,3)
    base_model = MobileNetV2(weights='imagenet',include_top=False,pooling=None,input_shape=input_shape)
    
    X = base_model.output
    
    X_Conv = Conv2D(3,(3,3,),strides=(1,1),padding='same', name='upsampled_conv',kernel_initializer=glorot_uniform(seed=0))(X)
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same', name = 'autoencoder/Upsample1')(X_Conv)
    X_feat1 = base_model.get_layer('block_13_expand_relu').output
    X_feat1 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat1_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat1)
    f1.append(X_feat1)
    f2.append(X_up)
    X_up = Add()([X_feat1,X_up]) 
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample2')(X_up)
    X_feat2 = base_model.get_layer('block_6_expand_relu').output
    X_feat2 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat2_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat2)
    f1.append(X_feat2)
    f2.append(X_up)
    X_up = Add()([X_feat2,X_up])    
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample3')(X_up)
    X_feat3 = base_model.get_layer('block_3_expand_relu').output
    X_feat3 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat3_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat3)
    f1.append(X_feat3)
    f2.append(X_up)
    X_up = Add()([X_feat3,X_up])
    X_up = Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample4')(X_up)
    X_feat4 = base_model.get_layer('block_1_expand_relu').output
    X_feat4 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat4_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat4)
    f1.append(X_feat4)
    f2.append(X_up)
    X_up = Add()([X_feat4,X_up])
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder')(X_up)
    X = GlobalAveragePooling2D(name='global_avg_pool')(X)
    X = Dense(256, name='Dense_1')(X)
    X = Dense(num_of_classes, name='Dense_2')(X)
    X = Activation('softmax', name='classifier')(X)
    
    #fm1 = feature_matching_loss(f_1=f1,f_2=f2)
    model = Model(inputs=base_model.input, outputs=[X,X_up,X], name='')

    model.compile(Adam(lr=.0001), loss=['categorical_crossentropy','mean_squared_error',
                                        feature_matching_loss(f_1=f1,f_2=f2)], metrics={'classifier': 'accuracy', 'autoencoder': 'mse'})
    
    model.summary()
    
    return model


#####################
#### OPTIC-NET ######
#####################

def res_conv(X, filters, base, s):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    X= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(2,2), strides=(s,s), padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)
    
    ##### Branch2 ####
    X_shortcut = BatchNormalization(axis=-1, name=name_base + '2/bn_1')(X_shortcut)
    X_shortcut= Activation('relu', name=name_base + '2/relu_1')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=name_base + '2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X

def res_identity(X, filters, base):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    Shortcut= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(Shortcut)

    # Second component BranchOut 1
    X1 = BatchNormalization(axis=-1, name=name_base + '1/ConvBn_2')(X)
    X1 = Activation('relu', name=name_base + '1/ConvRelu_2')(X1)
    X1 = Conv2D(filters=F2, kernel_size=(2,2), dilation_rate=(2, 2),strides=(1,1), padding='same', name=name_base + '1/Conv_2', kernel_initializer=glorot_uniform(seed=0))(X1)
    
    # Second component BrancOut 2
    X2 = BatchNormalization(axis=-1, name=name_base + '1/SepBn_2')(X)
    X2 = Activation('relu', name=name_base + '1/SepRelu_2')(X2)
    X2 = SeparableConv2D(filters=F2, kernel_size=(2,2), dilation_rate=(2, 2),strides=(1,1), padding='same', name=name_base + '1/SepConv_2', kernel_initializer=glorot_uniform(seed=0))(X2)
    
    # Second component Add-BranchOut
    X = Add(name=base + '/Add-2branches')([X1, X2])
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)    
    
    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '/Add')([X_shortcut,X])

    return X

def EncoderDecoder(X, name_base):
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name = name_base + '/Downsample1')(X)
    #X = Conv2D(outgoing_depth, (2,2), strides=(1,1), dilation_rate=(2,2), padding='same', name = name_base + '/DC1', kernel_initializer=glorot_uniform(seed=0))(X)    
    X = UpSampling2D(size=(2, 2),interpolation='bilinear',name = name_base + '/Upsample1')(X)
    X = Activation('sigmoid', name = name_base + '/Activate')(X)
    return X

def RDBI(X, filters, base, number):
    
    for i in range(number):
        X = res_identity(X, filters, base+ '/id_'+str(1+i))
    
    return X

def OpticNet_RFRL(input_size,num_of_classes):

    f1, f2 = [], []
    input_shape=(input_size, input_size, 3) # Height x Width x Channel
    X_input = Input(input_shape)

    X = Conv2D(64, (7,7), strides=(2,2), padding='same', name ='CONV1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=-1, name ='BN1')(X)
    X = Activation('relu', name ='RELU1')(X)
    
    X = res_conv(X, [64,64,256], 'RC0', 1)
    
    # MID 1
    
    X1 = EncoderDecoder(X, 'EncoderDecoder1')
    
    X2 = RDBI(X, [32,32,256], 'RDBI1',4)
    
    X = Multiply(name = 'Mutiply1')([X1,X2])
    
    X_feat_connection_3 = Add(name = 'Add1')([X,X1,X2])
    
    X = res_conv(X, [128,128,512], 'RC1', 2)
    
    # MID 2
    
    X1 = EncoderDecoder(X, 'EncoderDecoder2')
    
    X2 = RDBI(X, [64,64,512], 'RDBI2',4)
    
    X = Multiply(name = 'Mutiply2')([X1,X2])
    
    X_feat_connection_2 = Add(name = 'Add2')([X,X1,X2])
    
    X = res_conv(X, [256,256,1024], 'RC2', 2)
    
    # MID 3
    
    X1 = EncoderDecoder(X, 'EncoderDecoder3')
    
    X2 = RDBI(X, [128,128,1024], 'RDBI3',3)
    
    X = Multiply(name = 'Mutiply3')([X1,X2])
    
    X_feat_connection_1 = Add(name = 'Add3')([X,X1,X2])
    
    X = res_conv(X, [512,512,2048], 'RC3', 2)
    
    # MID 4
    
    X1 = EncoderDecoder(X, 'EncoderDecoder4')
    
    X2 = RDBI(X, [256,256,2048], 'RDBI4',3)
    
    X = Multiply(name = 'Mutiply4')([X1,X2])
    
    X = Add(name = 'Add4')([X,X1,X2])
    
    X_Conv = Conv2D(3,(3,3,),strides=(1,1),padding='same', name='upsampled_conv',kernel_initializer=glorot_uniform(seed=0))(X)
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample1')(X_Conv)
    X_feat1 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat1_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat_connection_1)
    f1.append(X_feat1)
    f2.append(X_up)
    X_up = Add()([X_feat1,X_up]) 
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample2')(X_up) 
    X_feat2 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat2_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat_connection_2)
    f1.append(X_feat2)
    f2.append(X_up)
    X_up = Add()([X_feat2,X_up])    
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder/Upsample3')(X_up)
    X_feat3 = Conv2D(3,(3,3),strides=(1,1),padding='same', name='feat3_conv',kernel_initializer=glorot_uniform(seed=0))(X_feat_connection_3)
    f1.append(X_feat3)
    f2.append(X_up)
    X_up = Add()([X_feat3,X_up])
    X_up =  Conv2DTranspose(3,(3,3,),strides=(2,2),padding='same',name = 'autoencoder')(X_up)

    
    X = GlobalAveragePooling2D(name='global_avg_pool')(X)
    X = Dense(256, name='Dense_1')(X)
    X = Dense(num_of_classes, name='Dense_2')(X)
    X = Activation('softmax', name='classifier')(X)

    
    #fm1 = feature_matching_loss(f_1=f1,f_2=f2)
    model = Model(inputs=X_input, outputs=[X,X_up,X], name='')

    model.compile(Adam(lr=.0001), loss=['categorical_crossentropy','mean_squared_error',
                                        feature_matching_loss(f_1=f1,f_2=f2)], metrics={'classifier': 'accuracy', 'autoencoder': 'mse'})
    
    model.summary()
    
    return model


