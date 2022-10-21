from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Conv2DTranspose, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad
import pickle
import tables


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, padding = 'same', activation = 'LeakyReLU'):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = padding)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation == 'sigmoid':
        x = Activation('sigmoid')(x)
        print('First Layer Sigmoid')
    else:
        x = LeakyReLU(alpha = 0.3)(x)
    
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = padding)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.3)(x)
    
    return x
  
def get_unet(input_img, n_filters = 10, dropout = 0.1, batchnorm = True):
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, activation = 'sigmoid')
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c5 = conv2d_block(p3, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
     
    u7 = Conv2DTranspose(n_filters * 4, (4, 4), strides = (2, 2), padding = 'same')(c5)
    u7 = Concatenate(axis=3)([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (4, 4), strides = (2, 2), padding = 'same')(c7)
    u8 = Concatenate(axis=3)([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (4, 4), strides = (2, 2), padding = 'same')(c8)
    u9 = Concatenate(axis=3)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(2, (1, 1),activation='linear')(c9) 
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


size = 25000
u_grid = 104
f_grid = 104
data_size = 0.8
test_size = 0.2
datatype = 'float32'
u_filename = "displacement_noise.h5"
f_filename = "tractions.h5"
batch_size = 132
f = tables.open_file(u_filename,mode='r')
ff = tables.open_file(f_filename,mode='r')
X_train = f.root.data[:int(size*data_size)]
Y_train = ff.root.data[:int(size*data_size)]
X_test = f.root.data[int(size*data_size):int(size*data_size)+int(size*test_size)]
Y_test = ff.root.data[int(size*data_size):int(size*data_size)+int(size*test_size)]

input_img = Input(shape=(u_grid, u_grid, 2))
unet = get_unet(input_img)

unet.compile(optimizer=Adagrad(), loss=MeanSquaredError())
print(unet.summary())

checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True)

history = unet.fit(x=X_train,y=Y_train,
                epochs=5000,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(X_test,Y_test),
                callbacks=[model_checkpoint_callback,TensorBoard(log_dir='logs')])

with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

unet.save("model.h5")
