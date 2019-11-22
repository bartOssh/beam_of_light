import torch
from .. import load_tensor_and_image_from_file, load_image_buffer_to_tensor
from ..params import COCO_INSTANCE_CATEGORY_NAMES as COCO


def find_boxes(output_predictions):
    """Sets boxed boundaries for predicted parameters

    Args:
        output_predictions (Tensor): AI model output predictions tensor

    Returns:
        prediction_boxed (object) Object with keys of predicted object names
        paired with boundary boxes of each prediction type
    """
    # todo: build Rust library for this iteration
    prediction_boxed = {}
    for x, t in enumerate(output_predictions.byte().cpu()):
        for y, v in enumerate(t):
            value = v.item()
            if value in prediction_boxed.keys():
                if prediction_boxed[value]['x_min'] > x:
                    prediction_boxed[value]['x_min'] = x
                if prediction_boxed[value]['y_min'] > y:
                    prediction_boxed[value]['y_min'] = y
                if prediction_boxed[value]['x_max'] < x:
                    prediction_boxed[value]['x_max'] = x
                if prediction_boxed[value]['y_max'] < y:
                    prediction_boxed[value]['y_max'] = y
            else:
                prediction_boxed[value] = {'x_min': x, 'y_min': y,
                                              'x_max': x, 'y_max': y}
    named = {}
    for k in prediction_boxed.keys():
        named[str(COCO[k])] = prediction_boxed[k]
    return named


def detect_local_file(args, device, nn_model):
    """Performs image recognition of local file

    Args:
        args (list): The list of arguments provided by the user
        device (object): The pytorch device object
        nn_model (object): The pytorch nn model instance that is pretrained
                            and evaluated

    Returns:
        tuple (input_image, output_predictions) Tuple of vectorized image
        to numpy array and AI model output predictions tensor
    """
    img_path = args.pretrained[0]
    input_batch, input_image = load_tensor_and_image_from_file(
        img_path, device
    )
    with torch.no_grad():
        output = nn_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return input_image, output_predictions


def detect_buffer(image_buf, device, nn_model):
    """Performs image recognition of given image buffer

    Args:
        image_buf (bytes buffer): The bytes buffer if image to recognize
        device (object): The pytorch device object
        nn_module (object): The pytorch nn model instance that is pretrained

    Returns:
        output_predictions: AI model output predictions tensor
    """
    input_batch = load_image_buffer_to_tensor(image_buf, device)
    with torch.no_grad():
        output = nn_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions
