import torch
from flask import Flask, request
from .. import YoloVisionRecognition
from .. import map_predictions_on_image_buffer


app = Flask(__name__)
yolo = YoloVisionRecognition('fcn_resnet101', 'cpu', True)


@app.route("/box", methods=['POST'])
def post_image_for_box():
    """
    Deep Learning module analise image and returns json info box
    """
    if request.content_type == "image/jpeg":
        predictions = yolo.detect_buffer(request.data)
        return YoloVisionRecognition.find_boxes(predictions)
    return 'Wrong Content-Type', 404


@app.route("/image", methods=['POST'])
def post_image_for_image():
    """
    Deep Learning analise image and returns recognition image
    """
    if request.content_type == "image/jpeg":
        predictions = yolo.detect_buffer(request.data)
        img = map_predictions_on_image_buffer(request.data, predictions)
        return img, 200
    return 'Wrong Content-Type', 404
