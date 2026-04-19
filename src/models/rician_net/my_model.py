from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers import Dropout, LeakyReLU,Input,add,Subtract
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
    # conv1
    x1 = identity_Block(inpt, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y1= Separ(inpt,16)
    x1 =  add([x1,y1])
    x2 = identity_Block(x1, nb_filter=16, kernel_size=(3, 3,3 ), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y2= Separ(y1,16)
    x2 =  add([x2,y2])
    x3 = identity_Block(x2, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y3= Separ(y2,16)
    x3 =  add([x3,y3])
    x4 = identity_Block(x3, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y4= Separ(y3,16)
    x4 =  add([x4,y4])
    x5 = identity_Block(x4, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y5 = Separ(y4, 16)
    x5 = add([x5, y5])    #x = attention(x, classes)
    x6 = identity_Block(x5, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y6 = Separ(y5, 16)
    x6 = add([x6, y6])
    x7 = identity_Block(x6, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(2,2,2), with_conv_shortcut=True)
    y7 = Separ(y6, 16)
    x7 = add([x7, y7])
    #x = attention(x, classes)
    x8 = identity_Block(x7, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
    y8 = Separ(y7, 16)
    x8 = add([x8, y8])
    x9 = identity_Block(x8, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y9 = Separ(y8, 16)
    x9 = add([x9, y9])
    #x = attention(x, classes)
    x10 = identity_Block(x9, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(2,2,2), with_conv_shortcut=True)
    y10 = Separ(y9, 16)
    x10 = add([x10, y10])
    x11 = identity_Block(x10, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
    y11 = Separ(y10, 16)
    x11 = add([x11, y11])
    x12 = identity_Block(x11, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y12 = Separ(y11, 16)
    x12 = add([x12, y12])
    x13 = identity_Block(x12, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(2,2,2), with_conv_shortcut=True)
    y13 = Separ(y12, 16)
    x13 = add([x13, y13])
    x14 = identity_Block(x13, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
    y14 = Separ(y13, 16)
    x14 = add([x14, y14])
    x15 = identity_Block(x14, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y15 = Separ(y14, 16)
    x15 = add([x15, y15])
    x16 = identity_Block(x15, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(2,2,2), with_conv_shortcut=True)
    y16 = Separ(y15, 16)
    x16 = add([x16, y16])
    x17 = identity_Block(x16, nb_filter=16, kernel_size=( 3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
    y17 = Separ(y16, 16)
    x17 = add([x17, y17])
    x18 = identity_Block(x17, nb_filter=16, kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
    y18 = Separ(y17, 16)
    x18 = add([x18, y18])
    x = Conv3D(1,kernel_size=(1,1,1))(x18)
    x = LeakyReLU( )(x)
    x= Subtract()([inpt,x])
    model = Model(inputs=inpt, outputs=x)
    return model