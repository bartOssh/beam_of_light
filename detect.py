import time
import torch
import argparse
from src import YoloVisionRecognition
from src import draw_image_and_recogintion, download_image

nn_model = 'deeplabv3_resnet101'
yolo = YoloVisionRecognition(nn_model, 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module allows to tests \
        deeplabv3_resnet101 neuron-network module')
    parser.add_argument('-p', '--pretrained', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what image to test against with \
                        pretrained weights')
    parser.add_argument('-t', '--train', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what image to test against with \
                        after training weights')
    parser.add_argument('-l', '--local', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what locally available and train \
                        model of nn we wont to use')
    parser.add_argument('-d', '--download', type=str, nargs=1,
                        metavar=('url'),
                        default=(None),
                        help='Allows to download image from the given url')
    args = parser.parse_args()
    if args.pretrained is not None and args.pretrained[0] is not None:
        ts_0 = time.time()
        input_image, output_predictions = yolo.recognize_local_file(
            args.pretrained[0]
            )
        ts_1 = time.time()
        print('\n RECOGNITION OBJECT: \n {} \n'
              .format(
                YoloVisionRecognition.find_boxes(output_predictions))
              )
        print('Total prediction time of the with model: {} took {} sec'
              .format(nn_model, int(ts_1 - ts_0)))
        draw_image_and_recogintion(input_image, output_predictions)
    if args.train is not None and args.train[0] is not None:
        # train(args, device)
        print('Not Implemented')
    if args.local is not None and args.local[0] is not None:
        print('Not Implemented')
    if args.download is not None and args.download[0] is not None:
        download_image(args.download)
