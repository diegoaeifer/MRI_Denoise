from keras.models import Model
from keras.layers.convolutional import Conv3D,SeparableConv2D
from keras.layers import Dropout, LeakyReLU,Input,add,Subtract,Reshape
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from sepconv3D import SeparableConv3D


def Separ(x,filters ):
    x1 = SeparableConv3D(filters,(3,3,3),padding='same')(x)
    x1 = LeakyReLU()(x1)
    x1 = SeparableConv3D(filters,(3,3,3),padding='same')(x1)
    x1 = LeakyReLU()(x1)
    y =  SeparableConv3D(filters,(1,1,1),padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    z = add([x1,y])
    return z
def Conv3d_BN(x, nb_filter, kernel_size, dilation_rate, padding='same'):
    x = Conv3D(nb_filter, kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
    y = BatchNormalization()(x)
    y = LeakyReLU()(y)
    return y
def identity_Block(inpt, nb_filter, kernel_size, dilation_rate, with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,dilation_rate=dilation_rate, padding='same')
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, dilation_rate=dilation_rate,kernel_size=kernel_size)
        x = Dropout(0.2)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def my_model(shape):
    inpt = Input(shape=shape)

    dilation_rates = [
        (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
        (2, 2, 2), (3, 3, 3), (1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 1, 1),
        (2, 2, 2), (3, 3, 3), (1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 1, 1)
    ]

    x = inpt
    y = inpt

    for dilation_rate in dilation_rates:
        x_next = identity_Block(x, nb_filter=16, kernel_size=(3, 3, 3), dilation_rate=dilation_rate, with_conv_shortcut=True)
        y_next = Separ(y, 16)
        x = add([x_next, y_next])
        y = y_next

    x = Conv3D(1, kernel_size=(1, 1, 1))(x)
    x = LeakyReLU()(x)
    x = Subtract()([inpt, x])

    model = Model(inputs=inpt, outputs=x)
    return model
