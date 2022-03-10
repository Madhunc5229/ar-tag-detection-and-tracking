# ar-tag-detection-and-tracking
#### This project will focus on detecting a custom AR Tag (a form offiducial marker), that is used for obtaining a point of reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely detection and tracking, both of which will be implemented in this project. The detection stage will involve finding the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in “view” throughout the sequence and performing image processing operations  such as superimposing another image on top of the AR Tag based on the tag’s orientation and position (a.k.a. the pose).


## Instructions to run the program:

Clone the files into your working directory

`git clone https://github.com/Madhunc5229/ar-tag-detection-and-tracking`

`cd ar-tag-detection-and-tracking`

## To detect the AR tag and decode the ID of the tag:

`python ARTag_detection.py`

## To track the AR tag thorughout the video and super impose it with an image:

`python ARTag_tracking.py`

