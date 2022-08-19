def MobileNet_RFRL(input_size,num_of_classes):    
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
