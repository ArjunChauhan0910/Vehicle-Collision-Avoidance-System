#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <cstring>
#include <ctime>
#include "samples_utility.hpp"

#include<iostream>

#define RESET   "\033[0m"
#define RED     "\033[1;31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW "\033[33m"  /* Yellow */

using namespace std;
using namespace cv;

CascadeClassifier car_cascade;

//void resetTracker(*cv::MultiTracker);

int main( int argc, const char **argv)
{
    const String keys =
    "{help h usage ? |      | Select Source File(-s) and Cascade file (-c)   }"
    "{c              |cars4.xml | Cascade   }"
    "{s              |file7.mp4     | source Input   }"
    ;


    CommandLineParser parser(argc, argv, keys);

    parser.about("ADAS v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    String car_cascade_name = parser.get<String>("c");
    
    if (! car_cascade.load(car_cascade_name))
    {
        cout <<"[ERROR] Cascade Not Found"<<endl;
    }

    VideoCapture capture; //declare a video capture object

    String camera_device = parser.get<String>("s");
    cout <<camera_device <<endl;
    capture.open(camera_device);

    if (!capture.isOpened())
    {
        cout <<"[ERROR] Video Not Found"<<endl;
    }

    Mat frame;
    Mat gray;
    Mat cropped;

    //create tracker pointer

    String trackingAlg = "KCF";
    MultiTracker trackers;
    vector<Rect2d> objects;
    std::vector<Ptr<Tracker> > algorithms;

    int frameCount = 0;

    while( capture.read(frame))
    {

        //Define ROI
        int rows = frame.rows;
        int cols = frame.cols;

        Point points[1][5];
        points[0][0] = Point( 0, rows);  //bottom left (x,y)
        points[0][1] = Point(0, 400); //top left (x,y)
        points[0][2] = Point( cols/8+100, 400); //top right (x,y)
        points[0][3] = Point(cols/8+100, 550);
        points[0][4] = Point(cols/8+400,rows); //bottom right (x,y)
        
        const Point* ppt[1] = {points[0]};
        int npt=5;
    
        
        if(frame.empty())
        {
            cout <<"[ERROR] No Frame Captured"<<endl;
        }


        if (frameCount%50 == 0)
        {
            //resetTracker(*trackers);
            MultiTracker *trackers;
            cout <<"Reset"<<endl;
        }


        if(frameCount%40 == 0) //check if 10 frames have passed
        {
            //cout <<"[INFO] 20 done"<<endl;
            Mat gray;
            Mat cropped;
            Mat mask = Mat::zeros(cvSize(cols, rows),CV_8UC1);

            objects.clear(); //let go of old tracks
            algorithms.clear();
            cout <<"Trackers "<<trackers.getObjects().size()<<endl;
            

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);

            fillPoly( mask,ppt, &npt,1,Scalar( 255, 255, 255 ));
            bitwise_and(gray, mask, cropped);

            imshow("Cropped", cropped);
            
            std::vector<Rect> cars;
            car_cascade.detectMultiScale(cropped, cars, 
                                    1.4,
                                    3,
                                    0,
                                    cvSize(50,50));
            for (size_t i = 0 ; i < cars.size(); i ++)
            {
                Point center(cars[i].x+cars[i].width/2, cars[i].y+cars[i].height/2);
                Point vertex1(cars[i].x, cars[i].y);
                Point vertex2(cars[i].x+cars[i].width, cars[i].y+cars[i].height);
                
                //show centroid and the bounding box of the car
                circle(frame, center, 1, Scalar(255,0,0), 2);
                rectangle(frame, vertex1, vertex2, Scalar(0,0,255),2);

                //create a bounding box and add it to the tracker              
                algorithms.push_back(createTrackerByName(trackingAlg));
                objects.push_back(cars[i]);
                trackers.add(algorithms,cropped,objects);

                //Check for hazard
                if (center.y < rows-70 && center.y > rows - 200)
                {
                    if (cars[i].y+cars[i].height > rows-70)
                    {
                        cout <<RED "[ALERT!] STOP!" RESET<<endl;
                    }

                    cout <<YELLOW "[WARNING] VEHICLE APPROACHING" RESET<<endl;
                }              
            }
        }

        
        bool ok = trackers.update(frame);
        //cout <<trackers.getObjects().size()<< endl;
        for(unsigned i=0;i<trackers.getObjects().size();i++)
                rectangle( frame, trackers.getObjects()[i], Scalar( 255, 255, 255 ), .5, 1 );

        /*if (ok)
        {   
            cout <<"IN"<<endl;

            for(unsigned i=0;i<trackers.getObjects().size();i++)
                rectangle( frame, trackers.getObjects()[i], Scalar( 255, 255, 255 ), 2, 1 );
        }*/

        /*bool ok = tracker->update(frame, bbox);
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }*/

        //fillPoly( frame,ppt, &npt,1,Scalar( 255, 255, 255 ));
        Point YellowStart = Point(0,rows-200);
        Point YellowEnd = Point(cols, rows-200);

        Point RedStart = Point(0, rows-70);
        Point RedEnd = Point(cols, rows-70);

        //line(frame, YellowStart, YellowEnd, Scalar(0,255,255), 1);
        //line(frame, RedStart, RedEnd, Scalar(0,0,255), 1);
        //polylines(frame, ppt, &npt, 1,true, Scalar(255,0,255), 0.5 );
        imshow("Detected", frame);

        frameCount++;
        
        if(waitKey(1) == 27)
        {
            break;
        }
    }

    return 0; 
}