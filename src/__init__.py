from .params import COCO_INSTANCE_CATEGORY_NAMES
from .utils import download_image, load_tensor_and_image, draw_image_and_recogintion
from .ai_callable.pretrained import detect, find_boxes
from .ai_callable.train import train
from .router import post_image_for_box, post_image_for_image

