import os
from skimage import measure
#from skimage.measure import structural_similarity as ssim #old
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow 
import glob

#from skimage.metrics import structural_similarity



def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	#print(imageA)
        #s = ssim(imageA, imageB) #old
	s = measure.compare_ssim(imageA, imageB, multichannel=True)
	print(s)
        # setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	#plt.show()
	return s


def unet(input_size=(256, 256, 3), num_classes=3):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

	# Example usage
	input_size = (256, 256, 3)
	num_classes = 3
	model = unet(input_size=input_size, num_classes=num_classes)
	model.summary()

diseaselist=os.listdir('static/Dataset')
print(diseaselist)
filename='c.jpeg'
ci=cv2.imread(filename)
gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
cv2.imwrite("static/Grayscale/"+filename,gray)
cv2.imshow("org",gray)
cv2.waitKey()

thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
cv2.imwrite("static/Threhold/"+filename,thresh)
cv2.imshow("org",thresh)
cv2.waitKey()

lower_green = np.array([34, 177, 76])
upper_green = np.array([255, 255, 255])
hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
binary = cv2.inRange(hsv_img, lower_green, upper_green)
cv2.imwrite("static/Binary/"+filename,gray)
cv2.imshow("org",binary)
cv2.waitKey()


'''       
width = 400
height = 400
dim = (width, height)
oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
flagger=1
for i in range(len(diseaselist)):
    if flagger==1:
        files = glob.glob('static/Dataset/'+diseaselist[i]+'/*')
        #print(len(files))
        for file in files:
            # resize image
            print(file)
            oi=cv2.imread(file)
            resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
            #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("comp",oresized)
            #cv2.waitKey()
            #cv2.imshow("org",resized)
            #cv2.waitKey()
            #ssim_score = structural_similarity(oresized, resized, multichannel=True)
            #print(ssim_score)
            ssimscore=compare_images(oresized, resized, "Comparison")
            if ssimscore>0.3:
                print(diseaselist[i])
                flagger=0
                break

'''






