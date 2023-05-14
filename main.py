import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
imgArrow = cv2.imread("arrow.png", cv2.IMREAD_UNCHANGED)
classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the bin images
imgBinList = []
pathFolderBins = "Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual

classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

# Initialize the variable to keep track of the last detected bin
lastBin = None

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread('background.png')

    prediction = classifier.getPrediction(img)
    print(prediction)
    classID = prediction[1]
    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

        # Check if the current bin is different from the last detected bin
        if classIDBin != lastBin:
            # Speak the name of the current bin
            binName = "recyclable" if classIDBin == 0 else \
                      "hazardous" if classIDBin == 1 else \
                      "food" if classIDBin == 2 else \
                      "residual"
            engine.say("Please dispose of the object in the " + binName + " bin.")
            engine.runAndWait()

            # Update the last detected bin to the current bin
            lastBin = classIDBin

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinList[classIDBin], (895, 374))

    imgBackground[148:148+340, 159:159+454] = imgResize
    # Displays
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)