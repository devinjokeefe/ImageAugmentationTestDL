from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
from PIL import Image

model = load_model('./model_no_augmentation.h5')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

img_1 = Image.fromarray(np.uint8(x_train[0, :, :, :])*255)
img_1.show()

x_test = x_test.astype('float32')
x_test /= 255

mean = np.mean(x_test, axis=(0,1,2,3))
std = np.std(x_test, axis=(0,1,2,3))
x_test = (x_test-mean)/(std+1e-7)

wrong_counter = 0
right_counter = 0

datagen = ImageDataGenerator (
    manipulation_frequency=0,
    manipulation_intensity=0,
    manipulation_color='all'
)

for i in range (x_test.shape[0]):
    image = x_test[i, :, :, :]
    print (image.shape)    
    img = Image.fromarray(np.uint8(image*255))
    img.show()
    
    image = np.expand_dims(image, axis=0)
    image_flow = datagen.flow(image)
    image_probs = np.zeros((10, 1))
        
    for n, new_images in enumerate(image_flow):
        image_probs = np.add(image_probs, model.predict(new_images))
        
        if n >= 2:
            if np.argmax(image_probs) != y_test[n, 0]:
                wrong_counter += 1
                print ("Wrong")
            else:
                right_counter += 1
                print ("Right")
            break
    
print ("Number of examples incorrect : {}".format(wrong_counter))
print ("Number of examples correct : {}".format(right_counter))
