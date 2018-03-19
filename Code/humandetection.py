# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
#Directories
main_path = os.path.dirname(os.path.realpath(__file__))[:-4]
run = "running/"
walk = "walking/"
#sources
data_path = main_path + "Dataset/"
run_path = data_path+run
walk_path = data_path+walk
#destinations
run_frames = main_path + "Frames/" + run
walk_frames = main_path + "Frames/" + walk
print(run_frames,walk_frames)


target_frames = walk_frames



img_paths = os.listdir(target_frames)
run_full=  [target_frames+e for e in img_paths]
#print(run_full)


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



ovrall =0
# loop over the image paths
for imagePath in run_full:
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width = 400)#min(400, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
    	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
    	cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
    	filename, len(rects), len(pick)))

    # show the output images
    #cv2.imshow("Before NMS", orig)
    if(len(pick)>0):
        cv2.imwrite(imagePath,image)
    else:
        os.remove(imagePath)
    #cv2.imshow("Found one!", orig)
    #cv2.waitKey(0)
    ovrall+=1
    print(ovrall)
