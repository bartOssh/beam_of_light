import torch
import numpy as np
from .. import load_tensor_and_image_from_file, load_image_buffer_to_tensor
from ..params import COCO_INSTANCE_CATEGORY_NAMES as COCO


recognition_nets = ['deeplabv3_resnet101', 'fcn_resnet101']
detection_nets = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169',
    'densenet201', 'googlenet', 'inception_v3', 'mobilenet_v2', 'resnet101',
    'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
    'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2',
    'wide_resnet50_2'
]


def get_available_models_names(force_reload=False):
    """Gives list of model names available on torch hub

    Return:
        models (list[string]): list of pytorch/vision models available
                                on pytorch hub
    """
    return torch.hub.list('pytorch/vision', force_reload=force_reload)


def check_if_new_models_in_hub():
    """Checks if there are new models available on torch hub

    Return:
        boolean: True if new models are available, False otherwise
    """
    hub_list = get_available_models_names(False)
    local_list = recognition_nets + detection_nets
    if len(set(hub_list) & set(local_list)) == len(local_list):
        for i in local_list:
            if i not in hub_list:
                return True
        return False
    return True


class YoloVision:
    """Trained deep neuron network vision interface

    Attributes:
        _models: Pretrained net models torch hub instance name
    """

    _models = 'pytorch/vision',

    def __init__(self, device='cpu'):
        """YoloVisionRecognition constructor

        Args:
            device (string): Name of device used for calculation
        """
        if device not in ['cuda', 'cpu']:
            raise Exception('Wrong device type passed to constructor')
        self._device = device


class YoloVisionRecognition(YoloVision):
    """Trained deep neuron network vision recognition interface
    """

    def __init__(self, nn_model=None, device='cpu'):
        """YoloVisionRecognition constructor

        Args:
            nn_module (string): Deep net vision model
            device (string): Name of device used for calculation
        """
        super().__init__(device)
        if nn_model not in recognition_nets:
            raise Exception('Wrong net model for recognition')
        self._nn_module = torch.hub.load('pytorch/vision', nn_model,
                                         pretrained=True).eval()

    @staticmethod
    def find_boxes(output_predictions):
        """Sets boxed boundaries for predicted parameters

        Args:
            output_predictions (Tensor): AI model output predictions tensor

        Returns:
            prediction_boxed (object) Object with keys of predicted object
                                        names paired with boundary
                                        boxes of each prediction type
        """
        prediction_boxed = {}
        for (x, y), t in np.ndenumerate(output_predictions
                                        .byte()
                                        .cpu()
                                        .numpy()):
            for _, v in np.ndenumerate(t):
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
                    prediction_boxed[value] = {
                        'x_min': x or 0,
                        'y_min': y or 0,
                        'x_max': x,
                        'y_max': y
                    }
        named = {}
        for k in prediction_boxed.keys():
            named[str(COCO[k])] = prediction_boxed[k]
        return named

    def recognize_local_file(self, image_path):
        """Performs image recognition of local file

        Args:
            image_path (string): path to image

        Returns:
            tuple (input_image, output_predictions) Vectorized image
                                                    to numpy array and AI model
                                                    output predictions tensor
        """
        input_batch, input_image = load_tensor_and_image_from_file(
            image_path, self._device
        )
        with torch.no_grad():
            output = self._nn_module(input_batch)
        output = output['out'][0]
        output_predictions = output.argmax(0)
        return input_image, output_predictions

    def recognize_buffer(self, image_buf):
        """Performs image recognition of given image buffer

        Args:
            image_buf (bytes buffer): The image to recognize

        Returns:
            output_predictions: AI model output predictions tensor
        """
        input_batch = load_image_buffer_to_tensor(image_buf, self._device)
        with torch.no_grad():
            output = self._nn_module(input_batch)
        output = output['out'][0]
        output_predictions = output.argmax(0)
        return output_predictions


class YoloVisionDetection(YoloVision):
    """Trained deep neuron network vision recognition interface
    """

    def __init__(self, nn_model=None, device='cpu'):
        """YoloVisionRecognition constructor

        Args:
            nn_module (string): Deep net vision model
            device (string): Name of device used for calculation
        """
        super().__init__(device)
        if nn_model not in detection_nets:
            raise Exception('Wrong net model for detection')
        self._nn_module = torch.hub.load('pytorch/vision', nn_model,
                                         pretrained=True).eval()

    @staticmethod
    def find_most_probable(output_predictions):
        """Looks in to output predictions for most probable detection

        Args:
            output_predictions: Tensor with probabilities of Yolo detection

        Returns:
            tuple (float, int): tuple of recognition probability and number
                                representing Yolo class of the probability
        """
        max_probability = torch.max(output_predictions)
        yolo_class_index = (output_predictions ==
                            max_probability).nonzero().item()
        return max_probability.item(), yolo_class_index

    def detect_local_file(self, image_path):
        """Performs image recognition of local file

        Args:
            image_path (string): path to image

        Returns:
            output_predictions: AI model output probabilities tensor of
                                of Yolo classes
        """
        input_batch, input_image = load_tensor_and_image_from_file(
            image_path, self._device
        )
        with torch.no_grad():
            output = self._nn_module(input_batch)
        output_predictions = torch.nn.functional.softmax(output[0], dim=0)
        return output_predictions

    def detect_buffer(self, image_buf):
        """Performs object detection on given image buffer

        Args:
            image_buf (bytes buffer): The image to recognize

        Returns:
            output_predictions: AI model output probabilities tensor
                                of Yolo classes
        """
        input_batch = load_image_buffer_to_tensor(image_buf, self._device)
        with torch.no_grad():
            output = self._nn_module(input_batch)
        output_predictions = torch.nn.functional.softmax(output[0], dim=0)
        return output_predictions
