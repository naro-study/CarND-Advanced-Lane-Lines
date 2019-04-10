import argparse
import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reference: Camera Calibration Tutorials
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
class CameraCalibration:
    def __init__(self,
                image_row=720,
                image_column=1280,
                corners_row=6,
                corners_column=9,
                is_visualize=False,
                is_save=False):
        self.image_row = image_row
        self.image_column = image_column
        self.corners_row = corners_row
        self.corners_column = corners_column
        self.is_visualize = is_visualize
        self.is_save = is_save

    def set_calibration_params(self):       
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.corners_column * self.corners_row,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.corners_column, 0:self.corners_row].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('./camera_cal/calibration*.jpg')
        print("Number of images: {}".format(len(images)))

        for idx, name in enumerate(images):
            img = cv2.imread(name)
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                grayscale_img, (self.corners_column, self.corners_row), None)
            
            print("idx number of current image is {}".format(idx))
            
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                cv2.drawChessboardCorners(img, (self.corners_column, self.corners_row), corners, ret)
                print("idx of detected image: {}".format(idx))

                # draw and display the corners
                if self.is_visualize == True:
                    cv2.imshow('draw corners image', img)
                    cv2.waitKey(500)

                # save corners detection images
                if self.is_save == True:
                    cv2.imwrite('./output_images/chessboard_corner_detection{}.jpg'.format(idx), img)

        if self.is_visualize == True:
            cv2.destroyAllWindows()

        # returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
        img_shape = (self.image_column, self.image_row)

        if len(objpoints) == len(imgpoints) and len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None) 
            print('success create calibration parameter')
        else:
            print('not enough points to calibrate')
        
        return mtx, dist
    
    def save_calibration_params(self, mtx, dist):
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open('./calibration.p', 'wb'))

    def load_calibration_params(self):
        dict = pickle.load(open('./calibration.p', mode='rb'))
        #print('dict={}'.format(dict))

        return dict

    def undistortion_image(self, image):
        dict = self.load_calibration_params()
        mtx = dict["mtx"]
        dist = dict["dist"]

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
                         
        return undistort_img
            
    def save_undistort_image(self):
        images = glob.glob('./camera_cal/calibration*.jpg')
        print("Number of images: {}".format(len(images)))

        for idx, name in enumerate(images):
            image = cv2.imread(name)
            undistort_img = self.undistortion_image(image=image)

            if self.is_visualize == True:
                #title = "undistortion image idx: {}".format(idx)
                #cv2.imshow(title, undistort_img)
                #cv2.waitKey(500)
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                ax1.imshow(image)
                ax1.set_title('Original Image', fontsize=24)
                ax2.imshow(undistort_img)
                ax2.set_title('Undistorted Image', fontsize=24)
                plt.show(block=True)

            if self.is_save == True:
                cv2.imwrite('./output_images/undistortion_image{}.jpg'.format(idx), undistort_img)
                #save_file_name = "undistorted_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
                #save_location = "./output_images/{}".format(save_file_name)
                #f.savefig(save_location, bbox_inches="tight")
        
        if self.is_visualize == True:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    cc = CameraCalibration(
        image_row=720,
        image_column=1280,
        corners_row=6,
        corners_column=9,
        is_visualize=True,
        is_save=False
    )

    #mtx, dist = cc.set_calibration_params()
    #cc.save_calibration_params(mtx, dist)
    cc.save_undistort_image()

        
