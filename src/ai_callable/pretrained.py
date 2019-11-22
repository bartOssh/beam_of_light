import torch
import numpy as np
from .. import load_tensor_and_image_from_file, load_image_buffer_to_tensor
from ..params import COCO_INSTANCE_CATEGORY_NAMES as COCO


class YoloVisionTrained:
    """Trained deep neuron network vision recognition interface

    Attributes:
        _models: Pretrained net models torch hub instance name
    """

    _models = 'pytorch/vision',

    def __init__(self, nn_model=None, device='cpu', force_reload=False):
        """YoloVisionTrained constructor

        Args:
            nn_module (string): Deep net vision model
            device (string): Name of device used for calculation
            forced_reaload (bool): True to reload entrypoints in cache
                                    False otherwise
        """
        if not isinstance(force_reload, bool):
            raise Exception('Wrong force reload parameter')
        self._entrypoints = self._get_entrypoints(force_reload)
        if device not in ['cuda', 'cpu']:
            raise Exception('Wrong device type passed to constructor')
        self._device = device
        if nn_model not in self._entrypoints:
            raise Exception('Passed nn_module to the \
                constructor don\'t exists')
        self._nn_module = torch.hub.load('pytorch/vision', nn_model,
            pretrained=True).eval()

    @staticmethod
    def get_names():
        """Returns list of available models

        Return:
            models (list[string]): list of pytorch/vision models available
                                    on pytorch hub
        """
        return self._get_entrypoints(force_reload=False)

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

    def detect_local_file(self, image_path):
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
            output = self._nn_module(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        return input_image, output_predictions

    def detect_buffer(self, image_buf):
        """Performs image recognition of given image buffer

        Args:
            image_buf (bytes buffer): The image to recognize

        Returns:
            output_predictions: AI model output predictions tensor
        """
        input_batch = load_image_buffer_to_tensor(image_buf, self._device)
        with torch.no_grad():
            output = self._nn_module(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        return output_predictions

    def _get_entrypoints(self, force_reload):
        return torch.hub.list(
            'pytorch/vision',
            force_reload=force_reload
            )
