import cv2 as cv

thres = 0.5

# img = cv.imread("../photos/lena.png")
cap = cv.VideoCapture(0)

cap.set(3,1080)
cap.set(4,1980)

classNames =[]
classFile = "coco.names"

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB (True)
while True:
    sucess,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold = 0.5)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img,box,color=(0,255,0),thickness=3)
            cv.putText(img,classNames[classId - 1],(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv.putText(img,str(round(confidence * 100,2)),(box[0]+200,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            


    cv.imshow("Lena",img)

    cv.waitKey(1)
