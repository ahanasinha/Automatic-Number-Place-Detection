import cv2
import numpy as np
import imutils
from pytesseract import pytesseract
import re
import smtplib

img = cv2.imread('./Pictures/Sample 2.jpg')

cv2.imshow('Original',img)


#********************************* Algorithm  1 *********************************************

#Gray Scaling
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
cv2.imshow('Edge',edged)

#Detecting the points of the contours from the edges and printing the points of the polygon closest to the shape of rectangle
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break


#Showing only the number plate image and turning rest of the background black
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

#Croping the Number plate image
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# Providing the tesseract executable
# location to pytesseract library
pytesseract.tesseract_cmd = r"C:\Users\Ahana Sinha\AppData\Local\Tesseract-OCR\tesseract.exe"

# Passing the image object to image_to_string() function
# This function will extract the text from the image
text = pytesseract.image_to_string(cropped_image)
text = ''.join(an for an in text if an.isalnum())
remove_lower = lambda text: re.sub('[a-z]', '', text)
text = remove_lower(text)



#*********************************Algorithm 2 **************************************


## Importing Haarcascade Algorithm data values
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

global read,plate
##Converting to Gray Scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detecting the Number Plate
nplate = cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in nplate:
    ## Cropping the Number Plate
    a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
    plate= img[y + a:y + h - a, x + b:x + w - b, :]


pytesseract.tesseract_cmd = r"C:\Users\Ahana Sinha\AppData\Local\Tesseract-OCR\tesseract.exe"
read = pytesseract.image_to_string(plate)
read=''.join(an for an in read if an.isalnum())
remove_lower = lambda text: re.sub('[a-z]', '', text)
read = remove_lower(read)


#********************************** Best Case Scenario ********************************

number=''
if read == text:
    print(text)
    number=text
    cv2.imshow('Plate',cropped_image)
    #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #plt.show()
else:
    if len(read)>len(text):
        print(read)
        number=read
        cv2.imshow('Plate',plate)
        #plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        #plt.show()
    else:
        print(text)
        number=text
        cv2.imshow('Plate',cropped_image)
        #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        #plt.show()


#***********************************E-mail Sending***************************************

sender_mail='anpr28project@gmail.com'
sender_psswrd='avsgrfhmjritrdxg'
reciever_mail='ahanasinha2003@gmail.com'

server = smtplib.SMTP('smtp.gmail.com',587)

server.starttls()

server.login(sender_mail,sender_psswrd)

message='A vehicle with '+number+' is detected.'

server.sendmail(sender_mail,reciever_mail,message)

print('E-mail sent')

#****************************************************************************

# user presses a key
cv2.waitKey(0)

# Destroying present windows on screen
cv2.destroyAllWindows()
