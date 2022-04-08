from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '64x3-CNN.model'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200, 200))

    # Preprocessing the image
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)
    #x = x.resize(x, (200, 200,))
    #x = x.reshape(-1, 200, 200, 3)

    IMG_SIZE = 200
    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE,))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds
   

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # pred_class = np.argmax(preds)
        result = str(pred_class[0][0][1])              # Convert to string
        return result

        if result == 0:
            print('Healthy')
        elif result == 1:
            print('Leprosy')
        else:
            print('Hyperthyroidism')
        
        return()


if __name__ == '__main__':
    app.run(debug=True)

