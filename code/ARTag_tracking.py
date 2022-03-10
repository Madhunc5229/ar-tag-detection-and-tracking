import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import linalg as LA
import math


def removePaperC(X,Y):
    x_min_index = X.index(min(X))
    x_max_index = X.index(max(X))

    y_min_index = Y.index(min(Y))
    y_max_index = Y.index(max(Y))

    X_min = X[x_min_index]
    Y_of_X_min = Y[x_min_index]

    X_max = X[x_max_index]
    Y_of_X_max = Y[x_max_index]

    X_of_Y_min = X[y_min_index]
    Y_min = Y[y_min_index]

    X_of_Y_max = X[y_max_index]
    Y_max = Y[y_max_index]

    X.remove(X_min)
    X.remove(X_max)

    if X_of_Y_min in X:
        X.remove(X_of_Y_min)

    if X_of_Y_max in X:
        X.remove(X_of_Y_max)

    Y.remove(Y_min)
    Y.remove(Y_max)

    if Y_of_X_max in Y:
        Y.remove(Y_of_X_max)
    if Y_of_X_min in Y:
        Y.remove(Y_of_X_min)

    return X,Y

def getCorners(X,Y):

    if (X.index(min(X)))>=0 and ((X.index(min(X)))< len(X)) and  ((0<=X.index(max(X)))) and ((X.index(min(X))< len(X))) and (Y.index(min(Y)))>=0 and ((Y.index(min(Y)))< len(Y)) and  ((0<=Y.index(max(Y)))) and ((Y.index(min(Y))< len(Y))):

        x_min_index = X.index(min(X))
        x_max_index = X.index(max(X))

        y_min_index = Y.index(min(Y))
        y_max_index = Y.index(max(Y))

        X_min = X[x_min_index]
        Y_of_X_min = Y[x_min_index]

        X_max = X[x_max_index]
        Y_of_X_max = Y[x_max_index]

        X_of_Y_min = X[y_min_index]
        Y_min = Y[y_min_index]

        X_of_Y_max = X[y_max_index]
        Y_max = Y[y_max_index]

    return [[X_min,Y_of_X_min],[X_of_Y_min,Y_min],[X_max,Y_of_X_max],[X_of_Y_max,Y_max]]


def getARtagID(tag_image):

    tag_corners_map = {}
    tag_corners_map["TL"] = tag_image[20:30, 20:30]
    tag_corners_map["TR"] = tag_image[20:30, 50:60] 
    tag_corners_map["BR"] = tag_image[50:60, 50:60] 
    tag_corners_map["BL"] = tag_image[50:60, 20:30] 

    inner_corners_map = {}
    inner_corners_map["TL"] = tag_image[30:40, 30:40]
    inner_corners_map["TR"] = tag_image[30:40, 40:50] 
    inner_corners_map["BR"] = tag_image[40:50, 40:50] 
    inner_corners_map["BL"] = tag_image[40:50, 30:40] 

    white_cell_corner = ''

    for cell_key in tag_corners_map:
        if is_cell_white(tag_corners_map[cell_key]):
            white_cell_corner = cell_key
            break
    if white_cell_corner == '':
        return None
    
    # print(white_cell_corner)
    id_number = [(is_cell_white(inner_corners_map[cell_key])) for cell_key in inner_corners_map]
    # print('id_number', id_number)
    re_orient_action_map = {'TR': [3, 0, 1, 2], 'TL': [2, 3, 0, 1], 'BL': [1, 2, 3, 0], 'BR': [0, 1, 2, 3]}
    
    new_corners_list = []
    tag_id = 0
    ortn = []
    for index, swap_ind in enumerate(re_orient_action_map[white_cell_corner]):
        # new_corners_list.append(corner_list[swap_ind])
        tag_id = tag_id + id_number[swap_ind]*math.pow(2, (index))
        ortn.append(swap_ind)
    return tag_id, ortn

def is_cell_white(cell):
    threshold = 200
    return 1 if (np.mean(cell) >= threshold) else 0

def calHomography(src_pts, dst_pts):

    X = np.array([[src_pts[0][0],src_pts[0][1],dst_pts[0][0],dst_pts[0][1]],[src_pts[1][0],src_pts[1][1],dst_pts[1][0],dst_pts[1][1]],[src_pts[2][0],src_pts[2][1],dst_pts[2][0],dst_pts[2][1]],[src_pts[3][0],src_pts[3][1],dst_pts[3][0],dst_pts[3][1]]])
    
    x = X[0:4,0]
    y = X[0:4,1]
    xp = X[0:4,2]
    yp = X[0:4,3]

    A_h = np.empty((8,9))

    #constructing A matrix
    for i in range(0,8,2):
        j = int(i/2)
        A_h[i] = [-x[j],-y[j],-1,0,0,0,(x[j]*xp[j]),(y[j]*xp[j]),xp[j]]

    for i in range(0,4):
        j = (2*i+1)
        A_h[j] = [0,0,0,-x[i],-y[i],-1,(x[i]*yp[i]),(y[i]*yp[i]),yp[i]] 


    u,s,v = LA.svd(A_h)


    # #The last vector in v is the solution for Ax=0
    X_h = (v[-1,:])

    return (np.reshape(X_h,(3,3)))/v[-1,-1]

def warping(dst_img, src_img, H):
    for i in range(dst_img.shape[0]):
        for j in range(dst_img.shape[1]):
            temp = np.matmul((LA.inv(H)),[i,j,1])
            temp /= temp[2]
            x, y, _ = temp.astype(int)
            if((x>=0) and (x<src_img.shape[1])) and (y>=0) and ((y<src_img.shape[0])):
                dst_img[j, i] = src_img[y, x]
    return dst_img

def fwdwarping(dst_img, src_img, H):
    for i in range(dst_img.shape[0]):
        for j in range(dst_img.shape[1]):
            temp = np.matmul((LA.inv(H)),[i,j,1])
            temp/= temp[2]
            x,y,_ = temp.astype(int)
            if((x>=0) and (x<src_img.shape[1])) and (y>=0) and ((y<src_img.shape[0])):
                src_img[y, x] = dst_img[j, i]
                src_img[y+1, x] = dst_img[j, i]
                src_img[y, x+1] = dst_img[j, i]
                src_img[y+1, x+1] = dst_img[j, i]
                src_img[y-1, x] = dst_img[j, i]
                src_img[y, x-1] = dst_img[j, i]
                src_img[y-1, x-1] = dst_img[j, i]

                src_img[y+2, x] = dst_img[j, i]
                src_img[y, x+2] = dst_img[j, i]
                src_img[y+2, x+2] = dst_img[j, i]
                src_img[y-2, x] = dst_img[j, i]
                src_img[y, x-2] = dst_img[j, i]
                src_img[y-2, x-2] = dst_img[j, i]
                
    return src_img


if __name__ == '__main__':
    cap = cv2.VideoCapture("1tagvideo.mp4")
    success = True
    frame_number = 0
    testudo_img = cv2.imread('testudo.png')


    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("testudo_on_tag.mp4", fourcc, 10, (640, 480))

    while success:
        success, frame = cap.read()

        #Converting image to gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #thresholding above 190 as 255
        _,frame_binary = cv2.threshold(gray_frame,190,255,cv2.THRESH_BINARY)

        # opening (eroding and dialating to remove blobs)
        frame_opend = cv2.morphologyEx(frame_binary, cv2.MORPH_OPEN, np.ones((5,5)))

        #FFT
        dft = cv2.dft(np.float32(frame_opend), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        #HPV
        rows, cols = frame_opend.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0


        # applying mask
        fshift = dft_shift * mask

        #IFFT
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        frame_fft = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        paper_corners = cv2.goodFeaturesToTrack(frame_fft, 20, 0.01, 50)

        paper_x_cord = []
        paper_y_cord = []


        # print(type(frame_corners))
        for i in paper_corners:
            x,y = i.ravel()
            # area.append(cv)
            paper_x_cord.append(x)
            paper_y_cord.append(y)

        X_new, Y_new = removePaperC(paper_x_cord,paper_y_cord)


        for corners in getCorners(X_new,Y_new):
            cv2.circle(frame, (corners[0], corners[1]), 10, (255, 0, 255), -1)
        Cs = getCorners(X_new,Y_new)


        src_pts = Cs
        dst_pts = [[0,80],[0,0],[80,0],[80,80]]

        H_of_tag = calHomography(src_pts,dst_pts)

        Tag = np.zeros((80,80), dtype=np.uint8)
        Tag = warping(Tag,frame_binary, H_of_tag)
        # opening (eroding and dialating to remove blobs)
        Tag_opend = cv2.morphologyEx(Tag, cv2.MORPH_OPEN, np.ones((3,3)))

        if getARtagID(Tag_opend) is None:
            continue

        Id, ortn = getARtagID(Tag_opend)

        testudo_pts = [[0,testudo_img.shape[0]],[0,0],[testudo_img.shape[1],0],[testudo_img.shape[0],testudo_img.shape[1]]]

        testudo_img = cv2.resize(testudo_img,(80,80))
        new_testudo_pts = [testudo_pts[ortn[3]],testudo_pts[ortn[0]],testudo_pts[ortn[1]],testudo_pts[ortn[2]]]

        # dst_pts = Cs
        # src_pts = new_testudo_pts
        dst_pts = new_testudo_pts

        H_of_test = calHomography(src_pts,dst_pts)
        testudo_on_frame = fwdwarping(testudo_img,frame ,H_of_test)

        cv2.imshow("Testudo on tag", testudo_on_frame)
        out.write(testudo_on_frame)
        # cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord('q'):
            break

        

