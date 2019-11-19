from flask import Flask, request

app = Flask(__name__)


@app.route("/box", methods=['POST'])
def post_image_for_box():
    """
    Deep Learnig module analize image and returns json info box
    """
    if request.content_type == "image/jpeg" and len(request.data) > 0:
        print(len(request.data))
        return 'Ok'
    return 'Wrong Content-Type', 404


@app.route("/image", methods=['POST'])
def post_image_for_image():
    """
    Deep Learnig analize image and returns recognition image
    """
    if request.content_type == "image/jpeg" and len(request.data) > 0:
        print(len(request.data))
        return 'Ok'
    return 'Wrong Content-Type', 404

