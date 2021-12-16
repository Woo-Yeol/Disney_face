from django.shortcuts import render
from django.conf import settings
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
# # Load the model
# model = load_model('keras_model.h5')

# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# # Replace this with the path to your image
# image = Image.open('<IMAGE_PATH>')
# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)

# #turn the image into a numpy array
# image_array = np.asarray(image)
# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# # Load the image into the array
# data[0] = normalized_image_array

# # run the inference
# prediction = model.predict(data)
# print(prediction)

BASE_DIR = getattr(settings, 'BASE_DIR', 'BASE_DIR')
Class = ['Anna', 'Ariel', 'Belle', 'Ruponzel', 'Elsa', 'Cinderella','Jasmine', 'Merida', 'Snow White', 'Arura', 'Tiana']

# Create your views here.
def home(request):
    if request.method == "POST":
        model = load_model(os.path.join(BASE_DIR,'keras_model.h5'))

        file = request.FILES['image']

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open(file)
        size = (224, 224)
        
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        
        prediction = model.predict(data)
        prediction = prediction.tolist()[0]

        class_index = prediction.index(max(prediction))
        return render(request,'home.html',{'prediction':Class[class_index]})

    return render(request,'home.html')