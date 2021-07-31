import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, train_size=7500, test_size=2500, random_state=9)

X_train_scaled = X_Train/255
X_test_scaled = X_Test/255

lr = LogisticRegression(solver="saga", multi_class='multinomial').fit(X_train_scaled, Y_Train)
test_predict = lr.predict(X_test_scaled)

Accuracy = accuracy_score(Y_Test, test_predict)

def getPrediction(image):
    Image_PIL = Image.open(image)

    Image_BW = Image_PIL.convert("L")
    Image_BW_RESIZE = Image_BW.resize((28, 28), Image.ANTIALIAS)
    
    Pixel_Filter = 20
    MIN_Pixel = np.percentile(Image_BW_RESIZE, Pixel_Filter)
    Image_BW_RESIZE_INVERTED_SCALED = np.clip(Image_BW_RESIZE-MIN_Pixel, 0, 255)

    MAX_Pixel = np.max(Image_BW_RESIZE)
    Image_BW_RESIZE_INVERTED_SCALED = np.asarray(Image_BW_RESIZE_INVERTED_SCALED)/MAX_Pixel

    test_sample = np.array(Image_BW_RESIZE_INVERTED_SCALED).reshape(1, 784)
    test_predict = lr.predict(test_sample)

    return test_predict[0]

