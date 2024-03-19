import cv2      # Image/video processing
import pyttsx3  # Text to speech
import cvzone   # Image/video processing, Image stacking
import math     
import numpy as np  # Arrays, Mathematical operations on images
from cvzone.ColorModule import ColorFinder  # Color detection
from image_cluster import ImageClusterer    # Cluster images based on color
import time

# Cluster model to analyze images
print('Please wait patiently while the clustering model is being trained...')
image_folder = "cashpics" 
num_clusters = 6
clusterer = ImageClusterer(num_clusters)  # Create instance
clusterer.train(image_folder)  # Train the clustering model
print('Thank you for waiting! The model has successfully been trained!')

# Color range (hue, saturation, and value)
# myColorFinder = ColorFinder(False)
# cOneHundred = {'hmin': 110, 'smin': 0, 'vmin': 82, 'hmax': 179, 'smax': 131, 'vmax': 198}
# cFifthy = {'hmin': 0, 'smin': 50, 'vmin': 112, 'hmax': 28, 'smax': 255, 'vmax': 255}
# cOneThousand = {'hmin': 59, 'smin': 0, 'vmin': 0, 'hmax': 113, 'smax': 255, 'vmax': 255}

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Settings window
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 1080, 300)

# Callback function
def empty(a):
    pass

# Settings window contents
cv2.createTrackbar("K Blur", "Settings", 19, 100, empty)
cv2.createTrackbar("Sigma Blur", "Settings", 4, 100, empty)
cv2.createTrackbar("Threshhold 1", "Settings", 86, 500, empty)
cv2.createTrackbar("Threshhold 2", "Settings", 19, 500, empty)
cv2.createTrackbar("Bill Size", "Settings", 100, 300, empty)

# Preprocess image
def preProcessing(img):

    # Retrieve values from settings
    kblur = cv2.getTrackbarPos("K Blur", "Settings")
    kblur = (math.ceil((math.ceil(kblur)/2)+0.5)*2)-1  # Round up to nearest odd
    sigmablur = cv2.getTrackbarPos("Sigma Blur", "Settings")
    edgethresh1 = cv2.getTrackbarPos("Threshhold 1", "Settings")
    edgethresh2 = cv2.getTrackbarPos("Threshhold 2", "Settings")

    # Gaussian blur
    imgBlur = cv2.GaussianBlur(img,(kblur,kblur),sigmablur)

    # Canny edge detection
    imgEdge = cv2.Canny(imgBlur,edgethresh1,edgethresh2)  # Get edges
    kernel = np.ones((4,4), np.uint8)
    imgEdge = cv2.dilate(imgEdge, kernel, iterations=1)           # Thicken edges
    imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel)  # Close edges

    # Return blurred image and edge-detected image
    return imgBlur, imgEdge 

# Image for displaying amount of money
moneyCount = np.zeros((400,600, 3), dtype = np.uint8)   # Create image
moneyCount.fill(255)              # Fill with white color
font = cv2.FONT_HERSHEY_SIMPLEX   # Choose a font
org = (20, 60)     # Determine the position to display the text
font_scale = 1     # Choose font scale and color
color = (0, 0, 0)  # Black color

previous_count = 0

# Camera loop
while True:
    # Reset money detected
    moneyAmt = 0

    # Capture and preprocess image
    success, img = cap.read()
    imgBlur, imgEdge = preProcessing(img)

    # Find contours from edge-detected image
    minArea = cv2.getTrackbarPos("Bill Size", "Settings")
    imgContours, conFound = cvzone.findContours(img, imgEdge, minArea=minArea)

    if conFound:
        moneyAmt = 0
        for contour in conFound:
            # Get number of sides (to check if it's a 4 sided bill)
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if( len(approx) <= 8 ): # Can be changed to 4 if there are no folds

                if( (contour['area']>3500) ): # Min area of money depends on distance

                    # Get box of image of individual money
                    x,y,w,h = contour['bbox']
                    imgCrop = img[y:y+h,x:x+w]

                    # Predict cluster
                    cluster, cashValue = clusterer.predict(imgCrop)
                    moneyAmt += int(cashValue)

        # Draw the text amount image
        moneyCount.fill(255)
        
        # Initialize text-to-speech
        #speech = pyttsx3.init()
        #voices = speech.getProperty('voices')
        #speech.setProperty('voice', voices[0].id)
        string_mon = str(moneyAmt)

        # Output text-to-speech
        if string_mon != '0' and previous_count != moneyAmt:
            print(f'Total Cash: {moneyAmt}')
            #speech.say(string_mon + 'Pesos')
            #speech.runAndWait()
            previous_count = moneyAmt

        # Display money counted
        cv2.putText(moneyCount, f'Php{moneyAmt}', org, font, font_scale, color, thickness=2, lineType=cv2.LINE_AA)

        # Define images to be displayed
        #imgStacked = cvzone.stackImages([img, moneyCount], 2, 1.4) # for showing camera and count only
        imgStacked = cvzone.stackImages([img, imgBlur, imgEdge, imgContours, moneyCount], 3, 1) # good for showing the process 
        
        # Resize images
        width = 720
        aspect_ratio = imgStacked.shape[1] / imgStacked.shape[0]
        height = int(width / aspect_ratio)
        resized_imgStacked = cv2.resize(imgStacked, (width, height))

        # Display images
        #cv2.imshow("imgColor", imgColor) # use for manual identifying color set "myColorFinder = ColorFinder(False)" to true
        cv2.imshow("BlindBills Prototype", resized_imgStacked)
        cv2.waitKey(1)

        # Delay scanning
        # time.sleep(5)
