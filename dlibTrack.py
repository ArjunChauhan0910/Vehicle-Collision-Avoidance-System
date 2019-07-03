#Import the OpenCV and dlib libraries
import cv2
import dlib

#Initialize a car cascade
carCascade = cv2.CascadeClassifier('cars4.xml')

def detectAndTrackMultiplecars():
    #Open the video/source
    capture = cv2.VideoCapture('file7.mp4')
    
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #The color of the rectangle we draw around the car
    rectangleColor = (0,0,255)

    #variables holding the current frame number and the current carID
    frameCounter = 0
    currentcarID = 0

    #Variables holding the correlation trackers and the name per carID
    carTrackers = {}
    carNames = {}

    try:
        while True:
            #Retrieve the latest image from the webcam

            rc,fullSizeBaseImage = capture.read()
            baseImage = fullSizeBaseImage

            #Resize the image to 320x240
            #baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('q'):
                break



            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest car
            resultImage = baseImage.copy()




            #STEPS:
            # * Update all trackers and remove the ones that are not 
            #   relevant anymore
            # * Every 10 frames:
            #       + Use car detection on the current frame and look
            #         for cars. 
            #       + For each found car, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new car-id


            #Increase the framecounter
            frameCounter += 1 



            #Update all the trackers and remove the ones for which the update
            #indicated the quality was not good enough
            cidsToDelete = []
            for cid in carTrackers.keys():
                trackingQuality = carTrackers[ cid ].update( baseImage )

                #If the tracking quality is not good enough, we must delete
                #this tracker
                if trackingQuality < 7:
                    cidsToDelete.append( cid )

            for cid in cidsToDelete:
                print("Removing cid " + str(cid) + " from list of trackers")
                carTrackers.pop( cid , None )




            #Every 20 frames, we will have to determine which cars
            #are present in the frame
            if (frameCounter % 10) == 0:



                #For the car detection, we need to make use of a gray
                #colored image so we will convert the baseImage to a
                #gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                cropped = gray[400:,:]
                cv2.imshow("Crop", cropped)
                #Now use the haar cascade detector to find all cars
                #in the image
                cars = carCascade.detectMultiScale(cropped,scaleFactor = 1.4, 
                                            minNeighbors = 3, 
                                            minSize=(50,50))



                #Loop over all cars and check if the area for this
                #car is the largest so far
                #We need to convert it to int here because of the
                #requirement of the dlib tracker. If we omit the cast to
                #int here, you will get cast errors since the detector
                #returns numpy.int32 and the tracker requires an int
                for (_x,_y,_w,_h) in cars:
                    x = int(_x)
                    y = int(_y)+400
                    w = int(_w)
                    h = int(_h)


                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h



                    #Variable holding information which carID we 
                    #matched with
                    matchedcid = None

                    #Now loop over all the trackers and check if the 
                    #centerpoint of the car is within the box of a 
                    #tracker
                    for cid in carTrackers.keys():
                        tracked_position =  carTrackers[cid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())


                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h


                        #check if the centerpoint of the car is within the 
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region 
                        #detected as a car. If both of these conditions hold
                        #we have a match
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                             ( t_y <= y_bar   <= (t_y + t_h)) and 
                             ( x   <= t_x_bar <= (x   + w  )) and 
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedcid = cid
                            cv2.circle(resultImage,(int(x_bar), int(y_bar)), 2, (255,0,255), -1)
                        
                        

                        #initiate warning system if car gets too close
                        if (gray.shape[:2][0] - 100 <y_bar < gray.shape[:2][0]):
                            print("[INFO] WARNING CAR APPROACHING")


                    #If no matched cid, then we have to create a new tracker
                    if matchedcid is None:

                        print("Creating new tracker " + str(currentcarID))

                        #Create and store the tracker 
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x,
                                                            y,
                                                            x+w,
                                                            y+h))

                        carTrackers[ currentcarID ] = tracker

                        currentcarID += 1




            #Now loop over all the trackers we have and draw the rectangle
            #around the detected cars. If we 'know' the name for this car
            #(i.e. the recognition thread is finished), we print the name
            #of the car, otherwise the message indicating we are detecting
            #the name of the car
            for cid in carTrackers.keys():
                tracked_position =  carTrackers[cid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)
            
            cv2.line(resultImage,(0,resultImage.shape[:2][0] - 100),(resultImage.shape[:2][1],resultImage.shape[:2][0]-100),(255,0,0),1)

            #Finally,show the images on the screen
            cv2.imshow("result-image", resultImage)


    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass

    #Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    detectAndTrackMultiplecars()
