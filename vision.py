#
# This is a demo program showing CameraServer usage with OpenCV to do image
# processing. The image is acquired from the USB camera, then a rectangle
# is put on the image and sent to the dashboard. OpenCV has many methods
# for different types of processing.
#
# NOTE: This code runs in its own process, so we cannot access the robot here,
#       nor can we create/use/see wpilib objects
#
# To try this code out locally (if you have robotpy-cscore installed), you
# can execute `python3 -m cscore vision.py:main`
#

import cv2
import numpy as np
import TargetFinder

from cscore import CameraServer

def main():
    cs = CameraServer.getInstance()
    cs.enableLogging()
    
    camera = cs.startAutomaticCapture()
    
    camera.setResolution(320, 240)
    
    # Get a CvSink. This will capture images from the camera
    cvSink = cs.getVideo()
    
    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cs.putVideo("Ultra cool vision", 320, 240)
    
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)    

    finder = TargetFinder.TargetFinder()

    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        time, img = cvSink.grabFrame(img)
        if time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError());
            # skip the rest of the current iteration
            continue
        
        # Put a rectangle on the image
        img = finder.processFrame(img, finder.boundaries)    
        # Give the output stream a new image to display
        outputStream.putFrame(img)