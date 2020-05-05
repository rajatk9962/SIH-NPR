


import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import os
#import shutil
#import cupy as cp
#from datetime import datetime
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
# from collections import Counter 
import collections



#-----------------------------
'''import pymysql

db = pymysql.connect("localhost", "root", "", "database1")
cursor = db.cursor()'''

#-----------------------------
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
license_list=['AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GA','GJ','HR','HP','JH','JK','KA','KL','LD','MH','ML','MN','MP','MZ','UP','NL','OD','PB','PY','RJ','SK','TN','TR','TS']
plates_text=[]

plate_image_dict = dict()
sdThresh = 20
font = cv.FONT_HERSHEY_SIMPLEX
counter_number=0
emptyframecounter=0
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold
inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image
framecount=0
resetcounter=0
d=0
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "classes.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "darknet-yolov3.cfg";
modelWeights = "lapi.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    #cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs,d):
    global resetcounter
    global counter_number
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    

    if resetcounter == -8:
        counter_number+=1
    else:
        counter_number =0
    resetcounter=-8
    for out in outs:
        
        # print("out.shape : ", out.shape)
        # print("Out of outs : ", len(out))
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            
            if detection[4]>confThreshold:
                print("DETECTION[4]: ", detection[4], " ---- SCORES----- ", scores[classId], " - THRESHOLD : ", confThreshold)
                # print("DETECTION : ",detection)
                print("Confidence :"+str(confidence))
                resetcounter = 1
            # else:
            #     if resetcounter!=1:
            #         resetcounter = -8
            #     else: 
            #         resetcounter = -3
            #     print('No confidence')

            if confidence > confThreshold:
   
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        framecrop = frame[top:top+height, left:left+width]
        #cv.imshow("crop",framecrop)
        croppedplate = "croppedplates/cropped_%d.jpg"%d
        
        cv.imwrite(croppedplate , framecrop.astype(np.uint8));
        
        temppath='temp/temp%d.png'%d  
        #img_dilation=np.array(img)         
        img = cv.imread(croppedplate, 0) 
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv.threshold(img, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        gray = cv.medianBlur(gray, 3)
        gray = cv.bilateralFilter(gray,9,75,75)
        cv.imwrite(temppath,gray)
        text = pytesseract.image_to_string(Image.open(temppath))
        text=text.upper()
        text=text.replace(" ","")
        text=text.replace("_","")
        text=re.sub(r'\W+', '', text)

        
        #d+=1
        # os.remove('temp.jpg')
        if text[:2] in license_list:
            if text[-1].isdigit() and text[2].isdigit():
                cp="cropped_%d.jpg"%d
                plate_image_dict[text] = cp
                # print(temppath+"   "+text)
                print("Detected Plate = "+text)
                plates_text.append(text)
                print("List of Plates Text : \n",plates_text)
        
        print(text)

            
# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (1024,768))

while cv.waitKey(1) < 0:

    # get frame from the video

    
    hasFrame, frame1 = cap.read()
    framecount+=1
    print("Frame Number = ",framecount)
    hasFrame, frame2 = cap.read()
    framecount+=1
    print("Frame Number = ",framecount)
    
    hasFrame, frame3 = cap.read()
    framecount+=1
    
    if frame1 is None or frame2 is None or frame3 is None :
        print("...............Frames completed.........")
        break

    frame1 = np.float32(frame1[768:1536,1024:2048])
    frame2 = np.float32(frame2[768:1536,1024:2048])
    

    print("Frame Number = ",framecount)
    frame3 = np.float32(frame3[768:1536,1024:2048])
    rows, cols, _ = np.shape(frame3)
    #cv.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, stDev = cv.meanStdDev(mod)
    
    #cv.imshow('dist', mod)
    #cv.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv.LINE_AA)
    if stDev > sdThresh:
        frames=[frame1,frame2,frame3]
        #print("Motion detected.. Do something!!!");
        # Stop the program if reached end of video
        for frame in frames:
            # print(frame)
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                cv.waitKey(3000)
                break
            #frame=(255-frame)
            
            # Stop the program if reached end of video
            # if not hasFrame:
            if counter_number>=15:
                # print(maximum+" inserted into DB")
                # cursor.execute(f"INSERT INTO table1(name) VALUES ('{maximum}')")
                # db.commit()

                counter_number=0
                if len(plates_text)>0:
                    print("Done processing !!!")
                    counter = collections.Counter(plates_text)
                    maximum = max(counter, key=counter.get)
                    print(counter)
                    
                    if maximum in plate_image_dict.keys(): 
                        print("YES PRESENT =", plate_image_dict[maximum]) 
                        print('car detected with number:'+maximum+"\n")
                        print("COUNTER NUMBER",counter_number)
                        '''shutil.move(r"C:\Users\rajat\Desktop\NPR\croppedplates\{}".format(plate_image_dict[maximum]),
                        r"C:\xampp\htdocs\ANPR_SLAVE\cropped_images\{}.jpg".format(maximum))
                        cursor.execute(f"INSERT INTO table1(name,image_name,status) VALUES ('{maximum}','{maximum}.jpg','notsure')")
                        db.commit()'''  
                    plates_text.clear()
                    plate_image_dict.clear()
                
                # print(plates_text)
                # print("RESET COUNTER",resetcounter)
                # print("Output file is stored as :  ", outputFile)
                # cv.waitKey(3000)
            elif not hasFrame:
                break

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))

            # Remove the bounding boxes with low confidence
            postprocess(frame, outs,d)

            d+=1
            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            '''if (args.image):
                cv.imwrite(outputFile, frame.astype(np.uint8));
            else:
                vid_writer.write(frame.astype(np.uint8))'''

    if cv.waitKey(1) & 0xFF == 27:
        break
