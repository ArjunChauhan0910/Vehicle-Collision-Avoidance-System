import cv2
import numpy as np
import matplotlib.pyplot as plt

print(">>> CAR DETECTOR V.0.1")

#declare array for counting number of cars
num_Cars = []
count = 0

#setup optical flow
 
##Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

##Parameters for Corner Detection 
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Create some random colors
color = np.random.randint(0,255,(100,3))

cascade_src = 'cars4.xml'
video_src = 'lane_project_video.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

#trying fast feature detector
#fast = cv2.FastFeatureDetector_create()

#create orb detector
#orb = cv2.ORB_create()

ret, img = cap.read()
mask = np.zeros_like(img)

found = False

while ret:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if (count%20 == 0):
        
        count = 0
        
        cars = car_cascade.detectMultiScale(gray, 
                                            scaleFactor = 1.2, 
                                            minNeighbors = 4, 
                                            minSize=(50,50))
        
        num_Cars.append(len(cars))

        if (len(cars) != 0):

            ft_points = []

            for (x, y, w, h) in cars:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                
                ft_points.append([x+w/2, y + h/2])

                found = True
                old_gray = gray

                p0 = cv2.goodFeaturesToTrack(old_gray[y:y+h,x:x+w], mask = None, **feature_params)

                #kp = fast.detect(gray[y:y+h,x:x+w], None)
                #imgC = img[y:y+h,x:x+w]
                #img = cv2.drawKeypoints(img[y:y+h,x:x+w], kp,img, color = (0,220,220))
    
    #FAST
    #kp = fast.detect(gray[y:y+h,x:x+w], None)
    #fast_im = cv2.drawKeypoints(img, kp,img[y:y+h, x:x+w], color = (0,220,220))

    '''
    Here's what you need to know.

    make sure the images are grayscale.
    your coordinate parameter that is i_old_pts should be single precision float meaning float32. This type is available in numpy use that. the float in python is float64
    the coordinate parameter i_old_pts(from your program) should be a numpy array with the dimension (n,1,2) where n represents the number of points.
    '''


#   ft_points = np.asarray(np.float32(ft_points))
#    ft_points.reshape(len(cars),1,2)

    if (found):
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray[y:y+h, x:x+w], gray[y:y+h,x:x+w], p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            a = int(a+x)
            b = int(b+y)
            c = int(c+x)
            d = int(d+y)
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            img = cv2.circle(img,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(img,mask)
    cv2.imshow('frame',img)



    #pts = np.asarray([kp[idx].pt for idx in range(0, len(kp))])#.reshape(-1, 1, 2)
    #print(pts[:][:][0][0])
    #print(type(pts))
    #print(pts[:][:][0][1])
    #img2 = cv2.drawKeypoints(img, kp,None, color = (0,220,220))

    #ORB
    #kp1 = orb.detect(img, None)
    #print(kp1.pt[0]+','+kp1.py[1])
    #kp1, des = orb.compute(img, kp)
    #img3 = cv2.drawKeypoints(img, kp1, None, color = (0,10,200), flags = 0)

    #cars = car_cascade.detectMultiScale(gray, 1.2,4,minSize=(50,50))

    #print(len(cars))
    #num_Cars.append(len(cars))'''


    '''for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        img2 = cv2.drawKeypoints(img, kp,None, color = (0,220,220))
        #p0 = np.asarray([x+w//2,y+h/2])'''

	
    cv2.imshow('video', img)
    #cv2.imshow("cropped", imgC)
    #cv2.imshow("FAST", fast_im)
    #cv2.imshow('FAST', img2)
    #cv2.imshow('ORB', img3)

    count += 1

    if cv2.waitKey(33) == 27:
        break

print(">> Video Over ")

plt.plot(num_Cars)
plt.show()


cv2.destroyAllWindows()



for (x,y,w,h) in cars:
    print("x",x)