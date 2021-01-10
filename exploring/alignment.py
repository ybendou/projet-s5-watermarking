
import cv2 
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RANSACRegressor

import pytesseract

class Preprocess():
    """
    Class for preprocessing an image, detecting edges and hough lines of the text and filtering them
    """
    def __init__(self):
        pass
    
    def remove_shadows(self,img):
        """
        Removes shadows from an image
        """
        rgb_planes = cv2.split(img) # split RGB colors 
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8)) # dilation
            bg_img = cv2.medianBlur(dilated_img, 21) # median bluring
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        result_ = cv2.merge(result_planes)
        return result_ 

    def edge_detection(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert to grayscale

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) # apply gaussian blur

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold) # Find edges using canny
        
        return edges
    
    def line_detection(self,edges):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        return lines
    
    def deskew(self,img):
        grayscale = rgb2gray(img)
        angle = determine_skew(grayscale)
        rotated = rotate(img, angle, resize=True) * 255
        print(f'Deskew angle : {np.round(angle,2)}')
        return angle,rotated.astype(np.uint8)
    
    def filter_hough_lines(self,lines,edges):
        """
        Detect Probabilistic Hough lines and drop lines with a small prjection on the x axis 
        """
        # removing small lines
        lines_tmp = []
        for line in lines : 
            for x1,y1,x2,y2 in line :
                if abs(x2-x1)>10: # we remove lines with small projection on the x axis (less than 10 pixels)
                    lines_tmp.append(np.array([[x1,y1,x2,y2]]))
        lines_tmp = np.array(lines_tmp)
        lines = lines_tmp.copy()
        return lines


    def compute_angles(self,lines):
        """
        Compute angles in degrees given lines
        """
        angles = []
        for line in lines : 
            for x1,y1,x2,y2 in line:
                ang = np.arctan((x1-x2)/(y2-y1)) # angle of rotation of the line (y2-y1 because the y axis is inverted (L-y))
                angles.append(ang)
        angles = np.array(angles)

        #converting angles to degree
        angles_deg = np.rad2deg(angles)
        return angles_deg


    def adjust_angles(self,angles_deg):
        """
        Adjusting the angles to between 0 and 90 degrees, we use the same logic as the one used in deskew library
        """
        # adjusting angles (from deskew repo)
        rot_angles = []
        for angle in angles_deg:
            if 0 <= angle <= 90:
                rot_angle = angle - 90
            elif -45 <= angle < 0:
                rot_angle = angle - 90
            elif -90 <= angle < -45:
                rot_angle = 90 + angle
            rot_angles.append(rot_angle)
        rot_angles = np.array(rot_angles)
        return rot_angles

    # filtering and only keeping lines with the same angle
    def filter_lines(self,lines,rot_angles,angle_desk):
        thresh = 5
        lines_r = lines.reshape(lines.shape[0],4)
        mask = (rot_angles>=angle_desk-abs(angle_desk))*(rot_angles<=angle_desk+abs(angle_desk)) # filtering the desired lines
        lines_candidates = lines_r[mask,:]

        lines_candidates = lines_candidates.reshape(lines_candidates.shape[0],1,lines_candidates.shape[1])
        return lines_candidates
    
    def remove_outliers(self,lines_candidates):
        """
        Removing outlier lines (lines far from the text) using their y coordinates
        """
        X_train = lines_candidates[:,:,3]
        clf = IsolationForest(max_samples=100,contamination=0.01) # consider 1% of the lines as outliers
        clf.fit(X_train)
        labels = clf.predict(X_train)

        lines_candidates = lines_candidates[labels!=-1] #removing outlier lines
        return lines_candidates

    def _compute_distance(self,l):
        """
        Compute the euclidian length of a line
        """
        return np.sqrt((l[0]-l[2])**2+(l[1]-l[3])**2)

    
    def get_longest_lines(self,lines_candidates,percentile_thresh=75):
        """
        Filter on the top longest lines, the percentage of the lines can be adjusted using percentile_thresh (100 - percentile_thresh)
        """
        distances = np.apply_along_axis(self._compute_distance, 1, lines_candidates.reshape(-1,4)) # compute length of the lines

        index_sorting = np.argsort(distances) #sort the index lines based on their distances 
        sorted_distances = np.array(sorted(distances)) # sort distances

        max_index = (sorted_distances[sorted_distances >= np.percentile(sorted_distances,percentile_thresh)]).shape[0] #take the top 25 longest lines
        longest_lines_indexs = index_sorting[-max_index:] # get longest lines indexes
        longest_lines = lines_candidates[longest_lines_indexs,:,:] # get longest lines
        return longest_lines

class Blind_image_adjustment():
    def __init__(self,image,lines,L):
        self.lines_candidates = lines
        self.lines_candidates_r = self._reshape_lines(self.lines_candidates)
        self.L = L
        self.image = image

    def _reshape_lines(self,lines):
        """
        Reshapes lines by dropping one axis
        """
        return lines.reshape(lines.shape[0],4)
    
    def find_perfect_rectangle(self):
        """
        Find a rectangle box from the given points
        """
        x = np.concatenate([self.lines_candidates_r[:,0],self.lines_candidates_r[:,2]])
        y = np.concatenate([self.lines_candidates_r[:,1],self.lines_candidates_r[:,3]])
        self.x_max      = x.max()
        self.x_min      = x.min()
        self.y_max      = y.max()
        self.y_min      = y.min()
        
        return self.x_max,self.x_min,self.y_max,self.y_min

    def _compute_line_equation(self,l):
        """
        Compute the line equation from 2 points of a line
        """
        x1,y1,x2,y2 = l
        y1 = self.L - y1 # inverse y axis to start from 0 to L 
        y2 = self.L - y2

        a = (y2-y1)/(x2-x1)
        b = y2 - x2*a
        return a,b

    def find_edge_line(self):
        lines = self.lines_candidates_r.copy()
        candidates = []
        
        
    def compute_edge_lines(self):
        """
        Find the edge lines surrounding the text, for each line we apply a different logic(index_dict)
        """
        greater = lambda a,b : a > b # returns a boolean value if a greater than b 
        smaller = lambda a,b : a < b # returns a boolean value if a smaller than b 
        
        # each type of line has some particular caracteristics which we define in the following dictionnary
        index_dict = {'right_vertical':{'coordinate_sorter_index':2, # x2 is maximum 
                                        'edge_coordinate_index':3, # leave all coordinates with y2 smaller
                                        'line_type':'vertical',
                                        'sort_logic':np.argmax,
                                        'mask':smaller},
                      'top_horizontal':{'coordinate_sorter_index':3, # y2 is minimal
                                        'edge_coordinate_index':2, # leave all coordinates with x2 smaller
                                        'line_type':'horizontal',
                                        'sort_logic':np.argmin,
                                        'mask':smaller},
                      'left_vertical':{'coordinate_sorter_index':0, # x1 is minimal
                                       'edge_coordinate_index':1, # leave all coordinates with y1 bigger
                                       'line_type':'vertical',
                                       'sort_logic':np.argmin,
                                       'mask':greater},
                      'low_horizontal':{'coordinate_sorter_index':1, # y1 is maximal 
                                        'edge_coordinate_index':0, # leave all coordinates with x1 bigger
                                        'line_type':'horizontal',
                                        'sort_logic':np.argmax,
                                        'mask':greater}
                    }
        
        edge_lines = {}
        for direction in index_dict.keys():
            candidates = []
            lines = self.lines_candidates_r.copy()
            coordinate_sorter_index = index_dict[direction]['coordinate_sorter_index']
            edge_coordinate_index   = index_dict[direction]['edge_coordinate_index']
            line_type               = index_dict[direction]['line_type']
            sorting_function        = index_dict[direction]['sort_logic']
            mask                    = index_dict[direction]['mask']
            while len(lines)>0: 
                first_line = lines[sorting_function(lines,axis=0)[coordinate_sorter_index],:] # follow the line that has the max coordinate
                candidates.append(first_line)
                
                lines = lines[mask(lines[:,edge_coordinate_index],first_line[edge_coordinate_index])]
            candidates = np.array(candidates).reshape(len(candidates),1,4)  
            edge_lines[direction] = self._fit_line(candidates[:,:,coordinate_sorter_index],
                                              candidates[:,:,edge_coordinate_index],
                                              line_type
                                             )
        return edge_lines
            
    def _fit_line(self,X,Y,line_type): 
        """
        Fits a robust line (robust to outliers) using RANSAC Regressor and returns two points from the line
        """
        model = RANSACRegressor()
        model.fit(X, Y)
        pred = model.predict(X).astype(int)

        if line_type == 'vertical':
            model_line = np.array([X[0][0],pred[0][0],X[-1][0],pred[-1][0]])
        elif line_type == 'horizontal':
            model_line = np.array([pred[0][0],X[0][0],pred[-1][0],X[-1][0]])
        else : 
            raise ValueError("Argument line_type only takes the values 'horizontal' and 'vertical'")
        
        return model_line
    
    
    def _compute_intersection(self,b1,b2,a1,a2):
        """
        Computes the cartesian intersection coordinates of two lines given their slope and bias
        """
        x_inter = (b1-b2)/(a2-a1)
        y_inter = self.L - int(a2*x_inter + b2)
        x_inter = int(x_inter)
        return x_inter,y_inter

    def find_corners(self,edge_lines):
        """
        Finds the intersection of the 4 surrounding lines (corner points)
        """
        a_m_lh,b_m_lh = self._compute_line_equation(edge_lines['low_horizontal']) # low horizontal line
        a_m_lv,b_m_lv = self._compute_line_equation(edge_lines['left_vertical']) # left vertical line
        a_m_rv,b_m_rv = self._compute_line_equation(edge_lines['right_vertical']) # right vertical line
        a_m_th,b_m_th = self._compute_line_equation(edge_lines['top_horizontal']) # top horizontal line


        x_top_left,y_top_left = self._compute_intersection(b_m_lv,b_m_th,a_m_lv,a_m_th)  
        x_low_left,y_low_left = self._compute_intersection(b_m_lv,b_m_lh,a_m_lv,a_m_lh)  
        x_top_right,y_top_right = self._compute_intersection(b_m_rv,b_m_th,a_m_rv,a_m_th) 
        x_low_right,y_low_right = self._compute_intersection(b_m_rv,b_m_lh,a_m_rv,a_m_lh) 
        self.corners = [x_top_left,y_top_left,   
                        x_low_left,y_low_left,   
                        x_top_right,y_top_right,  
                        x_low_right,y_low_right]
        return self.corners        
        
    def adjust_image(self,buffer=100):
        """
        Adjust a line using the homography transformation
        """
        dest = np.array([(self.x_min,self.y_min),
                         (self.x_min,self.y_max),
                         (self.x_max,self.y_min),
                         (self.x_max,self.y_max)
                        ])
        x_top_left, y_top_left, x_low_left, y_low_left, x_top_right, y_top_right, x_low_right,y_low_right = self.corners
        
        src =  np.array([(x_top_left,y_top_left),
                 (x_low_left,y_low_left),
                 (x_top_right,y_top_right),
                 (x_low_right,y_low_right)
                ])
        
        h, status = cv2.findHomography(src, dest,cv2.RANSAC, 5.0)
        
        x_border_min,x_border_max = (min(x_top_left,x_low_left,x_top_right,x_low_right)-buffer,
                                    max(x_top_left,x_low_left,x_top_right,x_low_right)+buffer)
        y_border_min,y_border_max = (min(y_top_left,y_low_left,y_top_right,y_low_right)-buffer,
                                    max(y_top_left,y_low_left,y_top_right,y_low_right)+buffer)
            
        im_out = cv2.warpPerspective(self.image[y_border_min:y_border_max,x_border_min:x_border_max],
                                     h,(2*buffer + self.x_max-self.x_min, 2*buffer + self.y_max-self.y_min))

        return im_out
        
        
        
        
        
        
        
        
        
        
        
        
        

        

