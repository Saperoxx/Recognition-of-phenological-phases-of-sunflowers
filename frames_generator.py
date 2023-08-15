import cv2
import os
# Read the video from specified path
video = cv2.VideoCapture("PATH_TO_VIDEO")
folder_name = "PATH_TO_FOLDER"
currentframe = 0
counter = 0
num_frames = 20

try:
# creating a folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

while(True):
# reading from frame
    ret,frame = video.read()

    if counter % num_frames == 0:
        if ret:
            # if video is still left continue creating images
            name = './' + folder_name + '/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    counter += 1

video.release()