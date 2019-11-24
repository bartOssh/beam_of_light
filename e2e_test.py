import io
import time
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


num_of_iter = 1
file = open('./assets/dog.jpeg', 'rb').read()
times = []

for i in range(0, num_of_iter):
    # testing endpoint box
    ts_start = time.time()
    res = requests.post(url='http://127.0.0.1:5000/box',
                        data=file,
                        headers={'Content-Type': 'image/jpeg'})
    print("\n RESPONSE POST to boxes, test num {} \n Sending buffer length:\
        {},\n Received {}"
          .format(i, len(file), res.json()))
    ts_stop = time.time()
    times.append(ts_stop - ts_start)

    # testing endpoint image
    ts_start = time.time()
    res = requests.post(url='http://127.0.0.1:5000/image',
                        data=file,
                        headers={'Content-Type': 'image/jpeg'})
    print("\n RESPONSE POST to image, test num {} \n Sending buffer length:\
        {},\n Received {} with content buffer of {} Bytes"
          .format(i, len(file), res, len(res.content)))
    ts_stop = time.time()
    times.append(ts_stop - ts_start)
    if i == num_of_iter - 1:
        image_solution = Image.open(io.BytesIO(res.content))
        image_send = Image.open(io.BytesIO(file))
        fig = plt.figure(figsize=(12, 6))
        for i, img in enumerate([image_solution, image_send]):
            fig.add_subplot(1, 2, i + 1)
            plt.imshow(img)
        plt.show()

_mean = np.mean(times)
print('Average time to perform full recognition process took {} seconds'
      .format(int(_mean)))
print('Full table of all times: \n{}'.format(times))
