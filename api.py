#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Caio Emmanuel
"""

from keras.models import load_model
from PIL import Image
import numpy as np
from flasgger import Swagger
from keras.preprocessing import image

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)

classifier = load_model('./classifier.h5')

@app.route('/predict_animal', methods=['POST'])
def predict_digit():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    test_image = image.load_img(request.files['image'], target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        return('Dog')
    else:
        return('Cat')

if __name__ == '__main__':
    app.run()