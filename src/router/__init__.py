from flask import Flask, request
from .. import YoloVisionRecognition, YoloVisionDetection
from .. import map_predictions_on_image_buffer


app = Flask(__name__)
yolo_reconize = YoloVisionRecognition('fcn_resnet101', 'cpu')
yolo_detect = YoloVisionDetection('mobilenet_v2', 'cpu')


@app.route("/recognition/box", methods=['POST'])
def post_image_for_box():
    """
    Deep Learning module analise image and returns json info box
    """
    if request.content_type == "image/jpeg":
        predictions = yolo_reconize.recognize_buffer(request.data)
        return YoloVisionRecognition.find_boxes(predictions)
    return 'Wrong Content-Type', 404


@app.route("/recognition/image", methods=['POST'])
def post_image_for_image():
    """
    Deep Learning analise image and returns recognition image
    """
    if request.content_type == "image/jpeg":
        predictions = yolo_reconize.recognize_buffer(request.data)
        return map_predictions_on_image_buffer(request.data, predictions), 200
    return 'Wrong Content-Type', 404


@app.route("/detection/classes", methods=['POST'])
def post_image_for_yolo_classes():
    """
    Deep Learning module analise image and returns json info box
    """
    if request.content_type == "image/jpeg":
        yolo_probabilities = yolo_detect.detect_buffer(request.data)
        print(yolo_probabilities.shape)
        return {'not_set': 'yet'}
    return 'Wrong Content-Type', 404


@app.route("/detection/most_likely", methods=['POST'])
def post_image_for_most_probable():
    """
    Deep Learning module analise image and returns json info box
    """
    if request.content_type == "image/jpeg":
        yolo_probabilities = yolo_detect.detect_buffer(request.data)
        max_probbility, yolo_class_index = YoloVisionDetection.find_most_probable(
            yolo_probabilities)
        print(max_probbility, yolo_class_index)
        return {'probability': max_probbility, 'yolo_class': yolo_class_index}, 200
    return 'Wrong Content-Type', 404
