# ar-tag-detection-and-tracking
#### This project will focus on detecting a custom AR Tag (a form offiducial marker), that is used for obtaining a point of reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely detection and tracking, both of which will be implemented in this project. The detection stage will involve finding the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in “view” throughout the sequence and performing image processing operations  such as superimposing another image on top of the AR Tag based on the tag’s orientation and position (a.k.a. the pose).


## Instructions to run the program:

`git clone https://github.com/Madhunc5229/ar-tag-detection-and-tracking`

`cd ar-tag-detection-and-tracking`

## To detect the AR tag and decode the ID of the tag:

`python ARTag_detection.py`

## To track the AR tag thorughout the video and super impose it with an image:

`python ARTag_tracking.py`

## AR Tag Detection:

![image](https://user-images.githubusercontent.com/61328094/157696518-982c24cf-7b5a-4764-bfa4-c25a68371fb9.png)

## AR Tag decoding:

![image](https://user-images.githubusercontent.com/61328094/157696653-120990a7-0912-44ec-8eb9-a55de6cf0559.png)

## Superimposing the testudo (University of Maryland's mascot, Go Terps!) on the AR Tag: 

![image](https://user-images.githubusercontent.com/61328094/157698513-89b09394-67e6-4ee2-92a1-2a02866d4588.png)

![image](https://user-images.githubusercontent.com/61328094/157698621-06b18c43-276c-4283-91c1-503243bd4655.png)

## Steps taken to detect the AR Tag:

• Capture the video, get a random frame from the video.  
• Convert the frame to gray image.  
• Apply threshold and convert the frame to binary image.  
• Performed morphology operation (erosion followed by dilation also known as opening) to get rid of all the noise and blobs.  
• Converted the frame into frequency domain using discrete Fourier transform function (cv2.dft()) and computed the magnitude spectrum of the dft frame.  
• Created a mask of same shape as frame with centre circular area as ‘0’ which acts as a high pass filter, so it allows only high frequency points tp pass.  
• Next, took inverse Fourier transform of the frame after multiplication with high pass filter.  
• This new frame generated after inverse Fourier transform has all the edges highlighted as white points and the background as black.  
• Passed the image after inverse Fourier Transform to GoodFeaturesToTrack(), which uses the Shi-Tomasi algorithm to detect corners.  
• Stored the output from GoodFeaturesToTrack() in a list and passed it to function ‘removePaperC()’ which removes the outer four corners I.e the paper corners and returns the list of remaining corners.  
• Next, this new list is to passed to ‘getCorners()’ function which gives the AR tag’s corners as output by calculating x minimum, y minimum , x maximum , y maximum and their corresponding corresponding cordinates.  

## Steps taken to decode the AR Tag:

• Stored the tag corners in a list and calculated Homography between the corner points and source points and destination points as four corner points of a new black image of shape called ‘Tag’ (80,80). 80 because it will be easy to divide while decoding the tag.  
• Computed the homography matrix by comptuing the solution for Ax = 0, where A is  

![image](https://user-images.githubusercontent.com/61328094/157697385-be88b53e-2482-4a05-97f1-97ce8863a720.png)

• Perform svd(A) and get u, s and v. Last column of v is considered as the solution for Ax=0. Reshaping the solution will give the Homograpghy matrix.  
• Image warping: For every point in the Tag, calculated the inverse of Homography matrix and multiplied with [x,y,1], where x and y are the indices of the Tag image, the output of the product contains [x’, y’, z’]. Therefore, normalized the values by dividing the elements by z’.  
• The values in frame for [x’/z’, y’/z’] will be copied to Tag[x,y].  
• Performed morphology for the output Tag to remove noise.  
• Passed the tag to getARtagID() to decode the tag.  
• Divided the tag into 8 X 8 squares by slicing separately for outer corners and inner corners.  
• Checked which outer square is white in order to get the orientation of the tag.  
• Defined the direction of the cycle of bits for each outer corner. That is, suppose the outer white corner is on Top right then the order is 3 , 0 , 1, 2.  
• After checking which outer corner is white, used the order assigned to that corner to check for inner squares, if the square is white, it is considered as 1 else 0.  
• After checking for the inner squares, converted the binary number to decimal and returned it as the tag ID. 

## Steps taken to super impose an image on the AR Tag:

• Carried out the same steps as previous question to get the AR tag corners, tag ID and the orientation of the tag.  
• Imported the testudo image and resized it to 80 x 80.  
• Computed homography between the corners of the testudo image and the tag corners and arranged these points according to the orientation of the tag. (i.e 3, 0, 1, 2).  
• Image warping: For every point in testudo image, multiplied the inverse of homography matrix with [x,y,1], where x and y are the pixel coordinates of the testudo image.  
• The product of above will contain [x’, y’, z’], to normalize dividing by z’. We get, [x’/z’,y’/z’]. Using these indices, assigned the value of testudo image to the frame.  
