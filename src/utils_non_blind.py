import cv2 
import numpy as np

class Preprocess():
    """
    Class for preprocessing an image by removing the shadows and rescaling it
    """
    def __init__(self, img, origin):
        self.img = img
        self.origin = origin
    
    def read_image(self):
        img = cv2.imread(img)
        origin = cv2.imread(origin)
        return img, origin
    
    def remove_shadows(self, img, origin):
        """
        Removes shadows from an image
        """
        rgb_planes = cv2.split(img) # split RGB colors 
        result_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8)) # dilation
            bg_img = cv2.medianBlur(dilated_img, 21) # median bluring
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        result_ = cv2.merge(result_planes)
        return result_ 

    def rescaling(self,img,origin,scale_percent = 40):
        """
        Rescaling the image
        """
        origin_unresized = origin.copy()
        width = int(img.shape[1]*scale_percent/100)
        height = int(img.shape[0]*scale_percent/100)
        dim = (width, height)
        img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        origin = cv2.resize(origin,dim, interpolation = cv2.INTER_AREA)
        return img, origin
    

class Image_adjustment():
    def __init__(self, img, origin):
        self.img = img
        self.origin = origin
        
    def detect_matches(self):
        """
        Detect matches between the image and the original document using the SIFT algorithm
        """
        
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp1 = orb.detect(self.img,None)
        # compute the descriptors with ORB
        kp1, des1 = orb.compute(self.img, kp1)
        kp2 = orb.detect(self.origin,None)
        kp2, des2 = orb.compute(self.origin, kp2)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches, kp1, kp2
    
    def align_image(self,matches,kp1,kp2):
        """
        Align the image using the result of sift algorithm
        """
        src=[]
        dest=[]
        for match in matches[:100]:
            kp_img1=kp1[match.queryIdx]
            kp_img2=kp2[match.trainIdx]
            src.append((kp_img1.pt[0],kp_img1.pt[1]))
            dest.append((kp_img2.pt[0],kp_img2.pt[1]))
        h, status = cv2.findHomography(np.array(src), np.array(dest),cv2.RANSAC, 5.0)
        im_out = cv2.warpPerspective(self.img,h,(self.origin.shape[1], self.origin.shape[0]))
        return im_out

        
        