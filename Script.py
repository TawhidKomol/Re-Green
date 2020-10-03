import cv2
import numpy as np

def image_masking(filepath):

	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 200
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format

	img = cv2.imread(filepath)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

	contour_info = []
	_, contours, __ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	for c in contours:
	    contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c),))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

	max_contour = contour_info[0]
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

	mask_stack = np.dstack([mask]*3)
	mask_stack  = mask_stack.astype('float32') / 255.0
	img = img.astype('float32') / 255.0

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
	masked = (masked * 255).astype('uint8')

	fileName, fileExtension = filepath.split('.')
	fileName += '-masked.'
	filepath = fileName + fileExtension
	print filepath

	cv2.imwrite(filepath, masked)

if __name__ == '__main__':
	filepath = raw_input("Enter Image File Name:\n")


import warnings
warnings.filterwarnings('ignore') # suppress import warnings

import os
import cv2
import tflearn
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

''' <global actions> '''

TRAIN_DIR = 'train/train'
TEST_DIR = 'test/test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dwij28leafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
tf.reset_default_graph()

''' </global actions> '''

def label_leaves(leaf):

    leaftype = leaf[0]
    ans = [0,0,0,0]

    if leaftype == 'h': ans = [1,0,0,0]
    elif leaftype == 'b': ans = [0,1,0,0]
    elif leaftype == 'v': ans = [0,0,1,0]
    elif leaftype == 'l': ans = [0,0,0,1]

    return ans

def create_training_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_leaves(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)

    return training_data

def main():

    train_data = create_training_data()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded')

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

if __name__ == '__main__': main()
 
import warnings
warnings.filterwarnings('ignore') # suppress import warnings

import os
import sys
import cv2
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

''' <global actions> '''

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dwij28leafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs

''' </global actions> '''

def process_verify_data(filepath):

	verifying_data = []

	img_name = filepath.split('.')[0]
	img = cv2.imread(filepath, cv2.IMREAD_COLOR)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	verifying_data = [np.array(img), img_name]

	np.save('verify_data.npy', verifying_data)

	return verifying_data

def analysis(filepath):

	verify_data = process_verify_data(filepath)

	str_label = "Cannot make a prediction."
	status = "Error"

	tf.reset_default_graph()

	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

	'''
	# relu:
	Relu is used in the middle / hidden layers of the network to regularize the activation.
	It is essentialy the function: max(0, x)
	Activation should not be in negative, either it should be zero or more than that.
	# softmax: 
	Softmax is used for the output layer in multi class classification problems.
	It is essentialy the function: log(1 + e^x)
	It outputs a vector of probabilities of each class.
	'''

	convnet = conv_2d(convnet, 32, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 128, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 32, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 3)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 4, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		model.load(MODEL_NAME)
		print 'Model loaded successfully.'
	else:
		print 'Error: Create a model using neural_network.py first.'

	img_data, img_name = verify_data[0], verify_data[1]

	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 0: str_label = 'Healthy'
	elif np.argmax(model_out) == 1: str_label = 'Bacterial'
	elif np.argmax(model_out) == 2: str_label = 'Viral'
	elif np.argmax(model_out) == 3: str_label = 'Lateblight'

	if str_label =='Healthy': status = 'Healthy'
	else: status = 'Unhealthy'

	result = 'Status: ' + status + '.'

	if (str_label != 'Healthy'): result += '\nDisease: ' + str_label + '.'

	return result

def main():
	filepath = raw_input("Enter Image File Name:\n")
	print analysis(filepath)

if __name__ == '__main__': main()
