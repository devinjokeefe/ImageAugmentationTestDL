import random
import numpy as np
import os
import scipy.misc

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(file):
    absFile = os.path.abspath("./cifar-10-batches-py/" + file)
    dict = unpickle(absFile)
    for key in dict.keys():
    	print(key)
    print("Unpacking {}".format(dict[b'batch_label']))
    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names

def visualize_image(X,Y,names):
    
    Y_vec = ['a' for i in range(10000)]
    file = open("./Y_Vec.txt", "w")
    for i in range (0, 10000):
        rgb = X[:,i]
        img = rgb.reshape(3,32,32).transpose([1, 2, 0])
        scipy.misc.imsave('./train_images/image{}.png'.format(i), img)
        Y_vec[i] = names[i]
        file.write(str(names[i]) + '\n')
    file.close()

X, Y, names = get_data("data_batch_1")
#X, Y, names = get_data("data_batch_2")
#X, Y, names = get_data("data_batch_3")
#X, Y, names = get_data("data_batch_4")
#X, Y, names = get_data("data_batch_5")

visualize_image(X, Y, names)
