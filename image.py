import cv2
import pytesseract
import re
import numpy as np
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata\mcr.traineddata"'
image = cv2.imread('C:/Users/mooham12314/Downloads/tesseract/img.jpg')
image = cv2.GaussianBlur(image,(13,13),0)
imagerotate = imutils.rotate(image, angle=347)
gray = cv2.cvtColor(imagerotate, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
kernel = np.ones((18,18), np.uint8) 
conv_img = np.array(thresh) 
erode = cv2.erode(conv_img, kernel, iterations=1)
cv2.imwrite('C:/Users/mooham12314/Downloads/tesseract/imgAfterPreProc222.jpg',erode)
print(pytesseract.get_languages(config=''))
text = pytesseract.image_to_string(erode,lang='micr', config="--psm 6")
text = re.sub('[^0-9]+', ' ', text)
print("Result : "+text)

#ref: https://stackoverflow.com/questions/66027978/python-number-recognition-on-colored-screen
