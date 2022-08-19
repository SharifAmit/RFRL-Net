from keras import callbacks

def callback_for_training(tf_log_dir_name='rfrl-network',patience_lr=10,snapshot_name=None):
    cb = [None] * 3
    """
    Tensorboard log callback
    """
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb[0]= tb
   
   
    
    """
    Model Checkpointer
    """
    checkpointer = callbacks.ModelCheckpoint(filepath=snapshot_name+".{epoch:02d}-{val_classifier_accuracy:.2f}.hdf5",
                            verbose=0,
                            monitor='val_classifier_accuracy')
    cb[1] = checkpointer
    
    """
    Reduce Learning Rate
    """
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_classifier_loss', factor=0.1, patience=8, verbose=1, min_lr=1e-8, mode='auto')
    cb[2] = reduce_lr_loss
    
    return cb
