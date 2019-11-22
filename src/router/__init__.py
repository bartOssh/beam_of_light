import torch
from flask import Flask, request
from .. import detect_buffer, find_boxes
from .. import map_predictions_on_image_buffer


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = torch.hub.load('pytorch/vision:v0.4.2', 'fcn_resnet101',
            pretrained=True).eval()


@app.route("/box", methods=['POST'])
def post_image_for_box():
    """
    Deep Learning module analise image and returns json info box
    """
    if request.content_type == "image/jpeg":
        predictions = detect_buffer(request.data, device, nn_model)
        return find_boxes(predictions)
    return 'Wrong Content-Type', 404


@app.route("/image", methods=['POST'])
def post_image_for_image():
    """
    Deep Learning analise image and returns recognition image
    """
    if request.content_type == "image/jpeg":
        predictions = detect_buffer(request.data, device, nn_model)
        return map_predictions_on_image_buffer(request.data, predictions)
    return 'Wrong Content-Type', 404
