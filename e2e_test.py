import requests

num_of_iter = 2

data = open('./assets/test_0.jpeg', 'rb').read()
for i in range(0, num_of_iter):
    res = requests.post(url='http://127.0.0.1:5000/box',
                        data=data,
                        headers={'Content-Type': 'image/jpeg'})
    print("\n RESPONSE POST to boxes, test num {} \n Sending buffer length: {},\n Received {}"
        .format(i, len(data), res.__dict__))
    res = requests.post(url='http://127.0.0.1:5000/image',
                        data=data,
                        headers={'Content-Type': 'image/jpeg'})
    print("\n RESPONSE POST to image, test num {} \n Sending buffer length: {},\n Received {}"
        .format(i, len(data), res))
