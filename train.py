#TODO Read in model configuration files

# TODO Dataset import and subsetting

# TODO Logging directories and TensorBoard monitoring

# TODO Training loop

import keras
import numpy as np

from keras.datasets import mnist

from architecture import VAE


def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Assuming square images here
    # Will need to be adapted for variable size images
    img_size = x_train.shape[1]
    input_shape = img_size * img_size
    x_train = np.reshape(x_train, [-1, input_shape])
    x_test = np.reshape(x_test, [-1, input_shape])
    
    # Normalize input vectors
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255


    vae = VAE(x_train)

if __name__ == '__main__':
    main()
