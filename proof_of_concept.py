#-------------------------------------------
# SEGMENT HAND REGION FROM A VIDEO SEQUENCE
#-------------------------------------------
import tensorflow as tf 
import h5py
import os
from time import sleep
import cv2
import numpy as np

# Load the model
model_path="C:/Users/34642/Downloads/AL-94.h5"
model=tf.keras.models.load_model(model_path)

# Global variables
bg = None

# To find the running average over the background
def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

# To segment the region of hand in the image
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

# Main function
if __name__ == "__main__":
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    
    # Move the region of interest (ROI)
    top, right, bottom, left = 150, 300, 300, 450
    
    num_frames = 0
    lista = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    while(True):
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        # Directly resize the ROI to 192x192
        resized = cv2.resize(roi, (192, 192), interpolation=cv2.INTER_AREA) / 255.0

        # Predict using the model
        pred = model.predict(resized.reshape(-1, 192, 192, 3))
        abc = 'ABCDEFGHIKLMNOPQRSTUVWXY'
        index = np.argsort(pred[0])

        # Get the top 3 predictions
        tres = index[-3:]
        l3 = abc[tres[0]]
        l2 = abc[tres[1]]
        l1 = abc[tres[2]]

        letra = abc[np.argmax(pred)]
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(clone, letra, (left - 90, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(clone, l2, (left - 150, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(clone, l3, (left - 10, top + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        num_frames += 1

        cv2.imshow("Video Feed", clone)

        keypress2 = cv2.waitKey(1)
        if keypress2 == ord(" "):
            letrica = lista[np.random.randint(24)]
            letraimagen = cv2.imread('Signos ASL/' + letrica)
            letraimagen = cv2.putText(letraimagen, str(letrica[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (226, 43, 138), 2, cv2.LINE_AA)
            cv2.imshow("Letra", letraimagen)

        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
