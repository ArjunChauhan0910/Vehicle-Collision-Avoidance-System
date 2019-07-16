# ADAS
This is a project to warn the driver about oncoming traffic. This was made keeping in mind the updated "rear view cameras" instead of mirrors in the next generation of cars. It gives out a warning to the driver asking them to watch out incase of incoming traffic.

## Dependencies 

* Python3
* OpenCV (3.4.4)
* Dlib 

## Approach 

There were 2 approaches used for this application. This is meant to be executed on a [embedded platform](https://www.imx6rex.com/) which is also catering to other applications being run on it simultaneously hence the approach had to be computationally efficient and fast at the same time. 

To cater to the requirements of the project, two pipelines were tested out. One pipeline was using HaarCascades in conjugation with KLOptical Flow algorithm and in the second one used HaarCascades with correlation tracker in DLib. 

In both approaches, the HaarCascade is run over a segment which is assumed to be relevant (i.e. the area relevant to the road.) every 10 frames so as to minimize the computational cost while serving its functionality. 

## Shortcomings

* HaarCascades are not accurate in detecting cars which are very close to the camera as the data that it has been trained on is limited in its perspective of the data.
* It is sensitive to occlusion, however, it is not an issue in this application since _any_ vehicle which is approaching you would be one to be aware of.


## Future Developments

* Need to collect appropriate training data for HaarCascades. The following sources have been explored [1](http://cogcomp.org/Data/Car/) [2](http://cbcl.mit.edu/software-datasets/CarData.html) [3](http://www-old.emt.tugraz.at/~pinz/data/GRAZ_02/)

* Training HaarCascades on different set of data which caters to our application (https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html, https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html)

* ~~Importing code to C/C++~~
