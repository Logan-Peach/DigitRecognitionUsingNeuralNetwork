import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model


class ImageLoader():
    def __init__(self, model = "digit-recognizer.h5", size = 28):
        self._model = load_model(model)
        self._size = size

    def preprocess(self, img_name):
        
        #input image
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        edges = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        imggg = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        while(type(img) == type(None)):
            img_name = str(input("Enter correct picture name and file format: "))
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            edges = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        rows = img.shape[0]
        cols = img.shape[1]

        #Loop through each pixel using thresholding to make the background black
        d = 30
        img = cv2.threshold(img, img.max()-d, 255, cv2.THRESH_TOZERO)[1]
        #Use canny edge detection
        edges = cv2.Canny(edges, img.max()-d, 255)
        #Add thresholded and edge images together
        img += edges
        #Use closing method to remove any gaps in the number
        kernel = np.ones((round(img.shape[0]*0.01), round(img.shape[1]*0.01)), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #Use a median blur to remove noise
        kSize = round(img.shape[0]*0.01)
        if kSize%2 == 0:
            kSize+=1
        #Pad the image to prepare it for filtering
        img = cv2.copyMakeBorder(img, kSize, kSize, kSize, kSize, cv2.BORDER_CONSTANT, None, 0)
        #Filter it
        img = cv2.medianBlur(img, kSize)
        #Remove padding
        for i in range(0, kSize):
            img = np.delete(img, 0, 0)
            img = np.delete(img, -1, 0)
            img = np.delete(img, 0, 1)
            img = np.delete(img, -1, 1)
        #Use dilate method to prevent loss of lines when resizing
        kernel = np.ones((round(img.shape[0]*0.01), round(img.shape[1]*0.01)), np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        #Resize image to proper dimensions to use with the model
        resized_img = cv2.resize(img, (self._size, self._size), interpolation = cv2.INTER_NEAREST)
        #Return the processed image
        return resized_img

    def predict(self, image):
        #Adjust image for use with the model
        adjusted_img = image.reshape(-1, self._size, self._size, 1)
        #Make the prediction
        prediction = self._model.predict([adjusted_img])
        #Print the results
        print(prediction)
        print(np.argmax(prediction[0]))

    def main(self):
        while True:
            imgname = str(input("Input an image name (or EXIT to quit): "))
            if imgname == "EXIT":
                exit()
                
            img = self.preprocess(imgname)
            self.predict(img)

            cv2.imshow("img", img)
            cv2.startWindowThread()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    IL = ImageLoader()
    IL.main()

main()
