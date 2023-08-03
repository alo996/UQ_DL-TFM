from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Conv2DTranspose, Concatenate, LeakyReLU
from keras.models import Model, load_model
from keras.losses import MeanSquaredError
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adagrad
from keras import mixed_precision
import tensorflow as tf
import pickle
import numpy as np
import h5py
import tqdm
from datetime import datetime


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, padding='same', activation='LeakyReLU'):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding=padding)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation == 'sigmoid':
        x = Activation('sigmoid')(x)
        print('First Layer Sigmoid')
    else:
        x = LeakyReLU(alpha=0.3)(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding=padding)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    return x


def get_unet(input_img, n_filters=10, dropout=0.1, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm, activation='sigmoid')
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c5 = conv2d_block(p3, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (4, 4), strides=(2, 2), padding='same')(c5)
    u7 = Concatenate(axis=3)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (4, 4), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate(axis=3)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (4, 4), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate(axis=3)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(2, (1, 1), activation='linear')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

'''
# Prepare training and validation dataset
# Train
np.random.seed(1)
tf.random.set_seed(1)
noise_percentage = 0.005

with tf.device("CPU"):
    dspl_train = np.array(h5py.File(
        '/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/allDisplacements.h5', 'r')['dspl'], dtype="float32")
    dspl_train_full = np.moveaxis(np.concatenate([dspl_train[i] for i in range(dspl_train.shape[0])], axis=0), 3, 1)
    del dspl_train

    sigma_bar = noise_percentage * np.mean(np.var(dspl_train_full, axis=(1, 2, 3)))
    print(f'percentage for noise level: {noise_percentage}')
    print(f'sigma_bar: {sigma_bar}')
    cov = [[sigma_bar, 0], [0, sigma_bar]]
    for i, x in tqdm.tqdm(enumerate(dspl_train_full), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl_train_full[i] = x + noise
    X_train = np.moveaxis(dspl_train_full, 1, 3)
    print(f'X_train_full.shape is {X_train.shape}')

    trac_train = np.array(
        h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/allTractions.h5','r')['trac'], dtype="float32")
    trac_train_full = np.moveaxis(np.concatenate([trac_train[i] for i in range(trac_train.shape[0])], axis=0), 3, 1)
    del trac_train
    Y_train = np.moveaxis(trac_train_full, 1, 3)
    print(f'Y_train_full.shape is {Y_train.shape}')


    dspl_val = np.array(h5py.File(
        '/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/allDisplacements.h5', 'r')['dspl'], dtype="float32")
    trac_val = np.array(
        h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/allTractions.h5',
                  'r')['trac'], dtype="float32")
    dspl_val_full = np.moveaxis(np.concatenate([dspl_val[i] for i in range(dspl_val.shape[0])], axis=0), 3, 1)
    del dspl_val
    trac_val_full = np.moveaxis(np.concatenate([trac_val[i] for i in range(trac_val.shape[0])], axis=0), 3, 1)
    del trac_val

    sigma_bar_val = noise_percentage * np.mean(np.var(dspl_val_full, axis=(1, 2, 3)))
    print(f'percentage for noise level: {noise_percentage}')
    print(f'sigma_bar_val: {sigma_bar_val}')
    cov = [[sigma_bar_val, 0], [0, sigma_bar_val]]
    for i, x in tqdm.tqdm(enumerate(dspl_val_full), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl_val_full[i] = x + noise
    X_val = np.moveaxis(dspl_val_full, 1, 3)
    Y_val = np.moveaxis(trac_val_full, 1, 3)
    print(f'_val.shape is {Y_val.shape}')

input_img = Input(shape=(104, 104, 2))
unet = get_unet(input_img)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

with tf.device("CPU"):
    train = tf.data.Dataset.from_tensor_slices((X_train, Y_train[:, :, :, 0:2]))
    train = train.shuffle(buffer_size=2 * 132)
    train = train.batch(batch_size=132)
    validate = tf.data.Dataset.from_tensor_slices((X_val, Y_val[:, :, :, 0:2])).batch(batch_size=132)

#loaded_model = load_model('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm/CNN_noisy-2023-Mar-21 16:57:44.h5')
unet.compile(optimizer=Adagrad(), loss=MeanSquaredError())
print(unet.summary())

#optimizer=tf.keras.optimizers.Adam(amsgrad=True, global_clipnorm=0.9, learning_rate=5e-3, weight_decay=5e-4)

checkpoint_filepath = 'CNN_noisy_final-{:%Y-%b-%d %H:%M:%S}_checkpoint.h5'.format(datetime.now())
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

#def scheduler(epoch, lr):
#    if epoch % 250 == 0 and epoch > 1:
#        print(f'new learning rate: {lr * 0.1}')
#        return lr * 0.1
#    else:
#        return lr

#scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = unet.fit(train,
                   epochs=5000,
                   batch_size=132,
                   shuffle=True,
                   validation_data=validate,
                   callbacks=[model_checkpoint_callback, TensorBoard(log_dir='logs'), early_stopping])

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

NAME = "CNN_noisy-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
unet.save(f'{NAME}.h5')
'''

