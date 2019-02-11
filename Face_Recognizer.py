# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Face Recognizer App
#
__author__ = "Oscar Castell"
__copyright__ = "Copyright 2017, Research Work"
__credits__ = ["Oscar Castell"]
__license__ = "GPL"
__version__ = "v3.0"
__maintainer__ = "Oscar Castell"
__email__ = "ocastell@xtec.cat"
__status__ = "Production"
import collections
#
# Import the Basic Libraries OpenCV
#
import os

import cv2
import cv2.face
import dlib
import numpy as np
#
# wxPython for the UI
#
import wx
import wx.xrc
from scipy.spatial import distance as dist
from wx.lib.buttons import GenButton
from wx.lib.pubsub import pub

config_dir = "./.Config/"
#
# OpenCV files haar_cascade
#
haar_cascade_file = config_dir + 'openCV/' + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_file)
#
# Create a HOG face detector using the built-in dlib class
#
predictor_model = config_dir + "dlib/" + "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)


colors = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))

MaxTakeImages = 20
(im_width, im_height) = (240, 240)
Default_Settings_File = "./.Config/Default_Settings.config"

FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))


def writeSettings(FileName, Data_Path, AspectRatio, VideoWidth, VideoHeight, Train_Method, MaxConfidence, yverbose):
    FileName = config_dir + FileName
    filename = open(FileName, "w+")
    filename.write(Data_Path + "\n")
    filename.write(AspectRatio + "\n")
    filename.write(str(VideoWidth) + "\n")
    filename.write(str(VideoHeight) + "\n")
    filename.write(Train_Method + "\n")
    filename.write(str(MaxConfidence) + "\n")
    if yverbose:
        Tverbose = "Output All"
    else:
        Tverbose = "Silent"
    filename.write(Tverbose)
    filename.close()


def loadSettings(DefaultSettingsFile=None):
    if DefaultSettingsFile is None: return
    Data_Path = None
    AspectRatio = None
    VideoWidth = None
    VideoHeight = None
    Train_Method = None
    haar_cascade_detector = False
    dlib_detector = False
    MaxConfidence = None
    yverbose = False
    filename = open(DefaultSettingsFile, "r")
    num_lin = 1
    for line in filename:
        if num_lin == 1:
            Data_Path = line.rstrip()
        elif num_lin == 2:
            AspectRatio = line.rstrip()
        elif num_lin == 3:
            VideoWidth = int(line.rstrip())
        elif num_lin == 4:
            VideoHeight = int(line.rstrip())
        elif num_lin == 5:
            Train_Method = line.rstrip()
            if Train_Method == "ANN":
                haar_cascade_detector = False
                dlib_detector = True
            else:
                haar_cascade_detector = True
                dlib_detector = False
        elif num_lin == 6:
            MaxConfidence = float(line.rstrip())
        elif num_lin == 7:
            verbose = line.rstrip()
            if verbose == "Silent":
                yverbose = False
            else:
                yverbose = True
        else:
            break
        num_lin = num_lin + 1
    filename.close()
    return Data_Path, AspectRatio, VideoWidth, VideoHeight, Train_Method, haar_cascade_detector, dlib_detector, \
           MaxConfidence, yverbose


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def roi_eyes(image, points, yverbose):
    left_eye = points[LEFT_EYE_POINTS]
    right_eye = points[RIGHT_EYE_POINTS]
    right_eye_hull = cv2.convexHull(right_eye)
    left_eye_hull = cv2.convexHull(left_eye)
    if yverbose:
        cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1)
    x_left_eye, y_left_eye, w_left_eye, h_left_eye = cv2.boundingRect(left_eye_hull)
    x_right_eye, y_right_eye, w_right_eye, h_right_eye = cv2.boundingRect(right_eye_hull)
    if yverbose:
        cv2.rectangle(image, (x_left_eye - 5, y_left_eye - 5),
                      (x_left_eye + w_left_eye + 5, y_left_eye + h_left_eye + 5), (0, 0, 255), 2)
        cv2.rectangle(image, (x_right_eye - 5, y_right_eye - 5),
                      (x_right_eye + w_right_eye + 5, y_right_eye + h_right_eye + 5), (0, 0, 255), 2)
    return left_eye_hull, right_eye_hull, x_left_eye, y_left_eye, w_left_eye, h_left_eye, x_right_eye, y_right_eye, \
           w_right_eye, h_right_eye


def getpupil_center(img, img2, yverbose):
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER_LEFT = 0
    TOTAL_LEFT = 0
    COUNTER_RIGHT = 0
    TOTAL_RIGHT = 0
    maxLoc = None
    detect = False
    img = cv2.medianBlur(img, 5)
    img_inv = cv2.bitwise_not(img)
    ret, img = cv2.threshold(img_inv, 100, 255, cv2.THRESH_TOZERO)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
    if maxLoc is not None:
        detect = True
        if yverbose:
            cv2.circle(img_inv, maxLoc, 1, 255, -1)
    return detect, maxLoc, img_inv


def detect_and_track_pupils(image=None, yverbose=False, nmesures=None, mesures=None, g=None):
    face = face_detector(image, 1)
    for i, face_rect in enumerate(face):
        x_pupil_right = 0
        y_pupil_right = 0
        x_pupil_left = 0
        y_pupil_left = 0
        pose_landmarks = face_pose_predictor(image, face_rect)
        points = shape_to_np(pose_landmarks)
        left_eye_hull, right_eye_hull, x_left_eye, y_left_eye, w_left_eye, h_left_eye, \
        x_right_eye, y_right_eye, w_right_eye, h_right_eye = roi_eyes(image, points, yverbose)
        mask = np.zeros_like(image)
        mask.fill(255)
        cv2.drawContours(mask, [left_eye_hull], -1, (0,0,0), -1)
        cv2.drawContours(mask, [right_eye_hull], -1, (0,0,0), -1)
        color = (0, 0, 255)
        #
        # Biometria
        #
        for (x, y) in points:
            cv2.circle(image, (x, y), 1, color, -1)
        #
        # Biometria: llarg ull esquerre
        #
        pt1 = (x1, y1) = points[36]
        pt2 = (x2, y2) = points[39]
        llarg_ull_esq = getdistance(pt1,pt2)
        color = (255, 255, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: llarg ull dret
        #
        pt1 = (x1, y1) = points[42]
        pt2 = (x2, y2) = points[45]
        llarg_ull_dret = getdistance(pt1,pt2)
        color = (255, 255, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: ample boca
        #
        pt1 = (x1, y1) = points[48]
        pt2 = (x2, y2) = points[54]
        ample_boca = getdistance(pt1,pt2)
        color = (255, 0, 255)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: llarg del nas
        #
        pt1 = (x1, y1) = points[27]
        pt2 = (x2, y2) = points[30]
        llarg_nas = getdistance(pt1,pt2)
        color = (255, 0, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: ample del nas
        #
        pt1 = (x1, y1) = points[31]
        pt2 = (x2, y2) = points[35]
        ample_nas = getdistance(pt1,pt2)
        color = (255, 0, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: llarg del rostre
        #
        pt1 = (x1, y1) = points[8]
        pt2 = (x2, y2) = points[27]
        llarg_rostre = getdistance(pt1,pt2)
        color = (0, 255, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        #
        # Biometria: ample del rostre
        #
        pt1 = (x1, y1) = points[0]
        pt2 = (x2, y2) = points[16]
        ample_rostre = getdistance(pt1,pt2)
        color = (0, 255, 0)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0)
        if (nmesures >= 100):
           white = (255,255,255)
           text = "Left Eye length: " + str(g['1']) + " mm"
           cv2.putText(image, text, (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Right Eye length: " + str(g['2']) + " mm"
           cv2.putText(image, text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Mouth length: " + str(g['3']) + " mm"
           cv2.putText(image, text, (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Nose length: " + str(g['4']) + " mm"
           cv2.putText(image, text, (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Nose breadth: " + str(g['5']) + " mm"
           cv2.putText(image, text, (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Face lenght: " + str(g['6']) + " mm"
           cv2.putText(image, text, (350, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "Face width: " + str(g['7']) + " mm"
           cv2.putText(image, text, (350, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
           text = "interpupillary distance: " + str(g['8']) + " mm"
           cv2.putText(image, text, (350, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
        #
        # Left eye
        #
        left_eye_roi = image[y_left_eye - 3:y_left_eye + h_left_eye + 3, x_left_eye - 3:x_left_eye + w_left_eye + 3]
        mask_eye = mask[y_left_eye - 3:y_left_eye + h_left_eye + 3, x_left_eye - 3:x_left_eye + w_left_eye + 3]
        out = np.zeros_like(left_eye_roi)
        out.fill(255)
        out[mask_eye == 0] = left_eye_roi[mask_eye == 0]
        out=cv2.GaussianBlur(out,(3,3),0,0,cv2.BORDER_DEFAULT)
        left_eye_roi_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        detect_left, maxLoc, img = getpupil_center(left_eye_roi_gray, left_eye_roi, yverbose)
        if yverbose:
            cv2.imshow("left eye inv", img)
        color = (255, 153, 51)
        if detect_left:
            (x, y) = maxLoc
            x_pupil_left =  x_left_eye + x - 3
            y_pupil_left =  y_left_eye + y - 3
            cv2.circle(image, (x_pupil_left, y_pupil_left), 4, color, -1)
        #
        # Right eye
        #
        right_eye_roi = image[y_right_eye-3:y_right_eye+h_right_eye+3,x_right_eye-3:x_right_eye+w_right_eye + 3]
        mask_eye = mask[y_right_eye-3:y_right_eye+h_right_eye+3,x_right_eye-3:x_right_eye+w_right_eye + 3]
        out = np.zeros_like(right_eye_roi)
        out.fill(255)
        out[mask_eye == 0] = right_eye_roi[mask_eye == 0]
        out=cv2.GaussianBlur(out,(3,3),0,0,cv2.BORDER_DEFAULT)
        right_eye_roi_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        detect_right, maxLoc, img = getpupil_center(right_eye_roi_gray, right_eye_roi, yverbose)
        if yverbose:
            cv2.imshow("right eye inv", img)
        if detect_right:
            (x, y) = maxLoc
            x_pupil_right = x_right_eye + x - 3
            y_pupil_right = y_right_eye + y - 3
            cv2.circle(image, (x_pupil_right, y_pupil_right), 4, color, -1)
        if detect_left and detect_right:
            inter_pupil_distance = getdistance((x_pupil_left,y_pupil_left),(x_pupil_right,y_pupil_right))
            mesures['ample_nas']=mesures['ample_nas']+ample_nas
            mesures['llarg_ull_esq']=mesures['llarg_ull_esq']+llarg_ull_esq
            mesures['llarg_ull_dret']=mesures['llarg_ull_dret']+llarg_ull_dret
            mesures['ample_boca']=mesures['ample_boca']+ample_boca
            mesures['llarg_nas']=mesures['llarg_nas']+llarg_nas
            mesures['llarg_rostre']=mesures['llarg_rostre']+llarg_rostre
            mesures['ample_rostre']=mesures['ample_rostre']+ample_rostre
            mesures['inter_pupil']=mesures['inter_pupil']+inter_pupil_distance
            if ((nmesures % 100) == 0 and nmesures > 0):
                g['1'] = float("{0:.2f}".format(mesures['llarg_ull_esq']/nmesures))
                g['2'] = float("{0:.2f}".format(mesures['llarg_ull_dret']/nmesures))
                g['3'] = float("{0:.2f}".format(mesures['ample_boca']/nmesures))
                g['4'] = float("{0:.2f}".format(mesures['llarg_nas']/nmesures))
                g['5'] = float("{0:.2f}".format(mesures['ample_nas']/nmesures))
                g['6'] = float("{0:.2f}".format(mesures['llarg_rostre']/nmesures))
                g['7'] = float("{0:.2f}".format(mesures['ample_rostre']/nmesures))
                g['8'] = float("{0:.2f}".format(mesures['inter_pupil']/nmesures))
            nmesures = nmesures + 1
            # Rectangle to be used with Subdiv2D
            h, w = image.shape[:2]
            rectan = (0, 0, h, w)
            # Create an instance of Subdiv2D
            subdiv = cv2.Subdiv2D(rectan)
            # Create an array of points.
            # Insert points into subdiv
            delauny = False
            for pepote in points:
                if pepote[0] < h and pepote[1] < w and pepote[0] >= 0 and pepote[1] >= 0:
                   p = (pepote[0], pepote[1])
                   subdiv.insert(p)
                else:
                    delauny = False
            # Draw delaunay triangles
            if delauny:
                draw_delaunay(image, subdiv, (255, 255, 255))
    return image, nmesures, mesures, g

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True


def detect_and_recognize_faces_dlib(image=None, names=None, ann=None, yverbose=False, MaxConfidence=None, prediction=None):
    face = face_detector(image, 1)
    ncol = 0
    for i, face_rect in enumerate(face):
        color = colors[ncol]
        (x, y, w, h) = dlib_to_cv2_rectangle(face_rect)
        pose_landmarks = face_pose_predictor(image, face_rect)
        points = shape_to_np(pose_landmarks)
        r = get_face_metric(points)
        test_matrix = [r]
        test_matrix = np.array(test_matrix, np.float32)
        _re, pred = ann.predict(test_matrix)
        index, confidence, _ = normalize(pred[0])
        sum = 0.0
        for num in prediction:
            sum += num
        if (abs(confidence)*100 >= MaxConfidence or sum > 0):
           prediction[index]=prediction[index]+abs(confidence)
           position = max(enumerate(prediction), key=lambda x: x[1])[0]
           text = names[position] + " " + "%0.2f" % confidence
           draw_rectangle(image, (x, y, w, h), color)
           draw_text(image, text, x, y, w, h, color)
           if yverbose:
               txt = "This person is " + text + " with a confidence of " + str(confidence)
               print (txt)
               print (pred)
           ncol = ncol + 1
    return


def detect_and_recognize_faces_opencv(image=None, face_recognizer=None, names=None, Train_Method=None,
                                      MaxConfidence=None, yverbose=None, NewW=None, NewH=None, prediction=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    ncol = 0
    for (x, y, w, h) in faces:
        img_test = gray[y:y + h, x:x + w]
        if Train_Method != "LBPH":
            img_test = cv2.resize(img_test, (NewW, NewH), cv2.INTER_LANCZOS4)
        label, confidence = face_recognizer.predict(img_test)
        #print confidence
        color = colors[ncol]
        draw_rectangle(image, (x, y, w, h), color)
        sum = 0.0
        for num in prediction:
            sum += num
        if sum > 0.0:
            position = max(enumerate(prediction), key=lambda x: x[1])[0]
            text = names[position]
        if confidence <= MaxConfidence:
            index = int(label)
            prediction[index]=prediction[index]+abs(confidence)
            position = max(enumerate(prediction), key=lambda x: x[1])[0]
            text = names[position]
        else:
            position = max(enumerate(prediction), key=lambda x: x[1])[0]
            text = names[position]
            #text = "Unknown"
        draw_text(image, text, x, y, w, h, color)
        if yverbose:
            txt = "This person is " + text + " with a confidence of " + str(confidence)
            print (txt)
        ncol = ncol + 1
    return


def normalize(vector):
    new_vector = np.array(vector)
    norm = np.linalg.norm(new_vector, ord=1)
    normed = new_vector / norm
    index = np.argmax(normed)
    confidence = normed[index]
    return index, confidence, normed


def getdistance(point_A=None, point_B=None):
    distance = dist.euclidean(point_A, point_B)
    return distance


def getmiddlepoint(point_A=None, point_B=None):
    a = np.array(point_A)
    b = np.array(point_B)
    middlepoint = (a + b) / 2
    return middlepoint[0], middlepoint[1]


def get_face_metric(points):
    width_left, width_right = points[0], points[16]
    top_left = points[18]
    top_right = points[25]
    bottom_left, bottom_right = points[50], points[52]
    top_average = int((top_left[1] + top_right[1]) / 2)
    bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
    coords = (width_left[0], width_right[0], top_average, bottom_average)
    corners = {'top_left': (coords[0], coords[2]),
               'bottom_left': (coords[0], coords[3]),
               'top_right': (coords[1], coords[2]),
               'bottom_right': (coords[1], coords[3])
               }
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    fwhr = float(width) / float(height)
    #
    # Extract features of the face
    #
    r = [float(getdistance(points[31], points[35])) / float(getdistance(points[33], points[27]))]
    r.append(float(getdistance(points[45], points[42])) / float(
        getdistance(getmiddlepoint(points[43], points[44]), getmiddlepoint(points[47], points[46]))))
    r.append(float(getdistance(points[36], points[39])) / float(
        getdistance(getmiddlepoint(points[37], points[38]), getmiddlepoint(points[41], points[40]))))
    r.append(float(getdistance(points[27], points[0])) / float(getdistance(points[33], points[27])))
    r.append(float(getdistance(points[27], points[16])) / float(getdistance(points[33], points[27])))
    r.append(float(getdistance(points[27], points[0])) / float(getdistance(points[31], points[36])))
    r.append(float(getdistance(points[27], points[16])) / float(getdistance(points[35], points[45])))
    r.append(float(getdistance(points[31], points[36])) / float(getdistance(points[31], points[27])))
    r.append(float(getdistance(points[45], points[35])) / float(getdistance(points[35], points[27])))
    r.append(fwhr)
    return r


def dlib_to_cv2_rectangle(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.right() - x
    h = face_rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def detect_face_dlib(image=None):
    roi_gray = None
    points = None
    r = None
    (x, y, w, h) = (None, None, None, None)
    detect = False
    color = (0, 0, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_detector(image, 1)
    for i, face_rect in enumerate(face):
        (x, y, w, h) = dlib_to_cv2_rectangle(face_rect)
        detect = True
        #
        # Get the the face's landmarks
        #
        pose_landmarks = face_pose_predictor(image, face_rect)
        points = shape_to_np(pose_landmarks)
        r = get_face_metric(points)
        #print r
        break
    if detect:
        draw_rectangle(image, (x, y, w, h), color)
        roi_gray = gray[y:y + h, x:x + w]
        aspectratio = float(h) / float(w)
        roi_gray = cv2.resize(roi_gray, (im_width, int(im_height * aspectratio)), cv2.INTER_LANCZOS4)
    return image, roi_gray, detect, points, r


def detect_face_opencv(image=None):
    roi_gray = None
    color = (0, 0, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(face) == 0:
        detect = False
        return image, gray, detect
    detect = True
    (x, y, w, h) = face[0]
    if detect:
        draw_rectangle(image, (x, y, w, h), color)
        roi_gray = gray[y:y + h, x:x + w]
        aspectratio = float(h) / float(w)
        roi_gray = cv2.resize(roi_gray, (im_width, int(im_height * aspectratio)), cv2.INTER_LANCZOS4)
    return image, roi_gray, detect


def draw_rectangle(img, rect, color=(0, 255, 0)):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def draw_text(img, text, x, y, w, h, color=(0, 255, 0)):
    cv2.putText(img, text, (x + 5, y + h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2)


def prepare_faces_data(Data_Path=None, yverbose=None, Train_Method=None):
    Pred_Freq = []
    labels = []
    faces = []
    samples = []
    num_samples = []
    data_folder_path = Data_Path
    dirs = os.listdir(data_folder_path)
    dirs2 = os.listdir(data_folder_path)
    for item in dirs:
        if item.startswith("."):
            dirs2.remove(item)
    dirs = dirs2
    nclase = 0
    nn = True
    NewH = 0
    NewW = 0
    for dir_name in dirs:
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        subject_images_names2 = subject_images_names
        for item in subject_images_names:
            if item.startswith("."):
                subject_images_names2.remove(item)
        subject_images_names = subject_images_names2
        nsamples = 0
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            filename, file_extension = os.path.splitext(image_path)
            if file_extension == ".info":
                filename = open(image_path, "r")
                characteristics_list = []
                for line in filename:
                    line = line.strip()
                    characteristics_list.append(float(line))
                filename.close()
                samples.append(characteristics_list)
                nsamples = nsamples + 1
                if yverbose:
                    print ("Name of data file : "), dir_name, image_path
                    print ("Adding image characteristics to inputs : ")
            elif file_extension == ".png":
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if nn:
                    NewH, NewW = image.shape[:2]
                    nn = False
                if Train_Method != "LBPH":
                    image = cv2.resize(image, (NewW, NewH), cv2.INTER_LANCZOS4)
                labels.append(dir_name)
                faces.append(image)
                if yverbose:
                    print ("Name, src: "), dir_name, image_path
                    cv2.imshow("Training on image...", image)
                    cv2.waitKey(100)
                    cv2.destroyAllWindows()
        num_samples.append(nsamples)
        nclase = nclase + 1

    names = []
    labels_num = []
    first_name = labels[0]
    num = 0
    names.append(first_name)
    for item in labels:
        if item == first_name:
            labels_num.append(num)
        else:
            names.append(item)
            first_name = item
            num = num + 1
            labels_num.append(num)

    targets = np.zeros((len(samples), len(names)), dtype=np.float32)
    ninput_layer = len(samples[0])
    nhidden_layer = int(1.5 * ninput_layer)
    noutput_layer = len(targets[0])
    layers = np.array([ninput_layer, nhidden_layer, noutput_layer], dtype=np.uint8)
    k = 0
    clase = 0
    for maxi in num_samples:
        for j in range(maxi):
            targets[k][clase] = 1.0
            k = k + 1
        clase = clase + 1
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(layers)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
    ann.setBackpropMomentumScale(0.1)
    ann.setBackpropWeightScale(0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 1.e-06))
    samples = np.array(samples, np.float32)
    targets = np.array(targets, np.float32)
    ann.train(samples, cv2.ml.ROW_SAMPLE, targets)
    file_model_name = "./Data/.ann_model"
    ann.save(file_model_name)
    for i in names:
        Pred_Freq.append(0.0)
    if yverbose:
        print ("Number of faces: "), len(names)
        print ("Number of images: "), len(labels_num)
        # counter = collections.Counter(labels_num)
        counter = collections.Counter()
        for item in counter.keys():
            txt = "Number of images for face " + names[int(item)] + " is " + str(counter.values()[int(item)])
            print (txt)
        print ("ANN - Number of input layers : "), ninput_layer
        print ("ANN - Number of hidden layers: "), nhidden_layer
        print ("ANN - Number of output layers: "), noutput_layer
        print ("ANN - Model saved in file    : "), file_model_name
        print ("ANN - Test of model          : "), file_model_name
        for i in range(len(labels_num) - 1):
            matrix_test = [samples[i, :]]
            test = np.array(matrix_test, np.float32)
            _ret, resp = ann.predict(test)
            index, confidence, _ = normalize(resp[0])
            txt = "         Vector test " + str(i) + " is : " + names[index]
            print (txt)
    return faces, labels_num, names, ann, NewW, NewH, Pred_Freq


def scale_image(img, MaxWidth, MaxHeight):
    H, W, channels = img.shape
    factor = float(MaxWidth) / float(W)
    NewW = MaxWidth
    NewH = int(float(H) * factor)
    if NewH > MaxHeight:
        factor = float(MaxHeight) / float(H)
        NewH = MaxHeight
        NewW = int(float(W) * factor)
    img_scaled = cv2.resize(img, (NewW, NewH), cv2.INTER_LANCZOS4)
    return img_scaled


###########################################################################
## Class DropTarget
###########################################################################

class DropTarget(wx.FileDropTarget):

    def __init__(self, widget, scale=True, videopath=False):
        wx.FileDropTarget.__init__(self)
        self.widget = widget
        self.scale = scale
        self.videopath = videopath

    def OnDropFiles(self, x, y, filenames):
        pathfile = filenames[0]
        pub.sendMessage('dnd', filepath=pathfile)
        return True


###########################################################################
## Class NewFaceFromVideoPanel
###########################################################################

class NewFaceFromVideoPanel(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Add Face from Video")
        self.panel = wx.Panel(self, wx.ID_ANY)

        global Default_Settings_File
        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)

        self.DataPath = self.Data_Path
        self.NumberImage = 0
        self.detect_face = False
        self.LiveVideoStream = False
        self.VideoFile = False
        self.capture = None
        self.frame = None
        self.height = None
        self.width = None
        self.bmp = None
        self.timex = None
        self.timex_photo = None
        self.detect_faces = False
        self.VideoFile = None
        self.directory = None
        self.points = None
        self.r = None
        self.roi_gray = None
        self.detect = False
        self.FaceImageName = None
        self.InfoFaceName = None
        self.img = None

        self.lblFaceName = wx.StaticText(self.panel, label="Face Name: ")
        self.lblErrorFaceName = wx.StaticText(self.panel)
        self.lblErrorFaceName.SetLabelMarkup(
            "<span foreground='red'>Attention: the name exists in Database. Try again.</span>")
        self.lblErrorFaceName.Hide()
        self.FaceName = wx.TextCtrl(self.panel, value="", size=(300, -1))
        self.BtnFaceName = GenButton(self.panel, label="Create")
        self.BtnFaceName.Bind(wx.EVT_BUTTON, self.onFaceName)

        self.firstLine = wx.StaticLine(self.panel, )

        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup("<span foreground='red'>Drag and Drop one Video or click on: </span>")
        self.BtnStartLiveVideo = GenButton(self.panel, label="Start Live Video Stream")
        color = (0, 255, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.BtnStartLiveVideo.Bind(wx.EVT_BUTTON, self.onStartLiveVideo)
        self.lblDragandDrop.Hide()
        self.BtnStartLiveVideo.Hide()
        pub.subscribe(self.update_image_on_dnd, 'dnd')
        img = wx.Image(self.VideoWidth, self.VideoHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, id=wx.ID_ANY, bitmap=wx.Bitmap(img), style=0)
        self.imageCtrl.Hide()
        filedroptarget = DropTarget(self, True, True)
        self.imageCtrl.SetDropTarget(filedroptarget)

        self.secondLine = wx.StaticLine(self.panel, )

        self.goButton = GenButton(self.panel, label="Start")
        color = (0, 255, 0)
        self.goButton.SetBackgroundColour(color)
        self.goButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")
        self.goButton.Bind(wx.EVT_BUTTON, self.onGo)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)

        self.updateSizer()
        self.Show()

    def updateSizer(self):
        topSizer = wx.BoxSizer(wx.VERTICAL)
        faceNameSizer = wx.BoxSizer(wx.HORIZONTAL)
        textErrorSizer = wx.BoxSizer(wx.HORIZONTAL)
        textImageSizer = wx.BoxSizer(wx.HORIZONTAL)
        ImageSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        faceNameSizer.Add(self.lblFaceName, 0, wx.ALL, 5)
        faceNameSizer.Add(self.FaceName, 1, wx.ALL | wx.EXPAND, 5)
        faceNameSizer.Add(self.BtnFaceName, 0, wx.ALL, 5)
        textErrorSizer.Add(self.lblErrorFaceName, 0, wx.ALL, 5)
        textImageSizer.Add(self.lblDragandDrop, 0, wx.ALL, 5)
        textImageSizer.Add(self.BtnStartLiveVideo, 0, wx.ALL, 5)
        ImageSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        btnSizer.Add(self.goButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)

        topSizer.Add(faceNameSizer, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(textImageSizer, 0, wx.CENTER)
        topSizer.Add(textErrorSizer, 0, wx.CENTER)
        topSizer.Add(ImageSizer, 0, wx.CENTER)
        topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)

    def onStartLiveVideo(self, event):
        if self.LiveVideoStream:
            self.LiveVideoStream = False
            self.detect_face = False
            self.goButton.Disable()
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex_photo.Stop()
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()
        else:
            self.VideoFile = False
            self.LiveVideoStream = True
            self.capture = cv2.VideoCapture(0)
            self.capture.set(3, 640)
            self.capture.set(4, 480)
            ret, self.frame = self.capture.read()
            if ret:
                self.height, self.width = self.frame.shape[:2]
                self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
                self.imageCtrl.SetBitmap(self.bmp)
                self.timex = wx.Timer(self)
                self.timex.Start(1000. / 24)
                self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
            else:
                self.LiveVideoStream = False
                self.detect_faces = False
                self.goButton.SetLabel("Start")
                color = (0, 255, 0)
                self.goButton.SetBackgroundColour(color)
                self.goButton.Disable()
                self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
                color = (0, 255, 0)
                self.BtnStartLiveVideo.SetBackgroundColour(color)
                self.timex.Stop()
                self.capture.release()
                img = wx.Image(self.VideoWidth, self.VideoHeight)
                self.imageCtrl.SetBitmap(wx.Bitmap(img))
                self.updateSizer()
                return
            self.goButton.Enable()
            color = (255, 0, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.BtnStartLiveVideo.SetLabel("Stop Live Video Stream")
        self.updateSizer()

    def StartVideo(self, filepath=None):
        self.VideoFile = False
        if filepath is None: return
        self.capture = cv2.VideoCapture(filepath)
        ret, self.frame = self.capture.read()
        if ret:
            self.VideoFile = True
            h, w = self.frame.shape[:2]
            aspectratio = float(self.VideoWidth) / float(w)
            self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
            self.imageCtrl.SetBitmap(self.bmp)
            self.timex = wx.Timer(self)
            self.timex.Start(1000. / 24)
            self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()
            return
        self.goButton.Enable()
        self.BtnStartLiveVideo.SetLabel("Stop Video Stream")
        color = (255, 0, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.LiveVideoStream = True
        self.updateSizer()

    def onFaceName(self, event):
        self.Name = str(self.FaceName.GetValue())
        if self.Name == "": return
        dirs = os.listdir(self.Data_Path)
        dirs2 = dirs
        for item in dirs:
            if item.startswith("."):
                dirs2.remove(item)
        dirs = dirs2
        for item in dirs:
            if item == self.Name:
                self.lblErrorFaceName.Show()
                self.updateSizer()
                return
        self.directory = self.Data_Path + "/" + self.Name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.lblErrorFaceName.Hide()
            self.lblDragandDrop.Show()
            self.BtnStartLiveVideo.Show()
            self.imageCtrl.Show()
            self.FaceName.Disable()
            self.BtnFaceName.Disable()
        self.updateSizer()

    def redraw(self, event):
        ret, self.frame = self.capture.read()
        if ret:
            if self.VideoFile:
                h, w = self.frame.shape[:2]
                aspectratio = float(self.VideoWidth) / float(w)
                self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.detect = False
            if self.detect_face:
               self.frame, self.roi_gray, self.detect, self.points, self.r = detect_face_dlib(image=self.frame)
               if self.detect and self.yverbose:
                  color = (255, 0, 0)
                  for (x, y) in self.points:
                      cv2.circle(self.frame, (x, y), 1, color, -1)
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
            self.bmp.CopyFromBuffer(self.frame)
            self.imageCtrl.SetBitmap(self.bmp)
            self.Refresh()
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()
            return

    def onCancel(self, event):
        self.Close()

    def update_image_on_dnd(self, filepath):
        self.StartVideo(filepath)

    def onGo(self, event):
        if self.detect_face:
            self.timex_photo.Stop()
            self.detect_face = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
        else:
            self.timex_photo = wx.Timer(self)
            interval = 2000
            self.timex_photo.Start(interval)
            self.Bind(wx.EVT_TIMER, self.take_photo, self.timex_photo)
            self.detect_face = True
            self.goButton.SetLabel("Stop")
            color = (255, 0, 0)
            self.goButton.SetBackgroundColour(color)

    def take_photo(self, event):
        if not self.detect: return
        if self.NumberImage > MaxTakeImages:
            self.timex_photo.Stop()
            self.detect_face = False
            self.goButton.SetLabel("Start")
            self.goButton.Disable()
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            return
        self.FaceImageName = self.directory + "/" + str(self.NumberImage) + ".png"
        self.InfoFaceName = self.directory + "/" + str(self.NumberImage) + ".info"
        filename = open(self.InfoFaceName, "w+")
        for item in self.r:
            filename.write("%s\n" % item)
        filename.close()
        cv2.imwrite(self.FaceImageName, self.roi_gray)
        self.NumberImage = self.NumberImage + 1


###########################################################################
#  Class PupilTrackingInVideoPanel
###########################################################################

class PupilTrackingInVideoPanel(wx.Dialog):

    def __init__(self, parent):
        global Default_Settings_File
        wx.Dialog.__init__(self, None, wx.ID_ANY, "Pupil tracking in Video")
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.detect_faces = False
        self.NumberImage = 0
        self.detect_face = False
        self.LiveVideoStream = False
        self.VideoFile = False
        self.capture = None
        self.frame = None
        self.height = None
        self.width = None
        self.bmp = None
        self.timex = None
        self.detect_faces = False
        self.VideoFile = None
        self.directory = None
        self.points = None
        self.r = None
        self.roi_gray = None
        self.detect = False
        self.FaceImageName = None
        self.pupil_detector = False
        self.image = self.frame
        self.nmesures = 0
        self.mesures = {'llarg_ull_esq':0.0,
                        'llarg_ull_dret':0.0,
                        'ample_boca':0.0,
                        'llarg_nas':0.0,
                        'ample_nas':0.0,
                        'llarg_rostre':0.0,
                        'ample_rostre':0.0,
                        'inter_pupil':0.0}
        self.g = {'1':0.0,
                  '2':0.0,
                  '3':0.0,
                  '4':0.0,
                  '5':0.0,
                  '6':0.0,
                  '7':0.0,
                  '8':0.0}

        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)
        title_text = "<span foreground='blue' font-size='250%'>Pupil Tracking Method: LBPH Face Recognizer</span>"
        imgText = "<span foreground='red' font-size='15pt'>Drag and Drop one Video or: </span>"
        self.lblTitle = wx.StaticText(self.panel)
        self.lblTitle.SetLabelMarkup(title_text)
        self.firstLine = wx.StaticLine(self.panel, )
        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup(imgText)
        self.BtnStartLiveVideo = GenButton(self.panel, label="Start Live Video Stream")
        color = (0, 255, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.BtnStartLiveVideo.Bind(wx.EVT_BUTTON, self.onStartLiveVideo)

        pub.subscribe(self.update_image_on_dnd, 'dnd')
        filedroptarget = DropTarget(self, False, True)

        img = wx.Image(self.VideoWidth, self.VideoHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, id=wx.ID_ANY, bitmap=wx.Bitmap(img))
        self.imageCtrl.SetDropTarget(filedroptarget)
        self.secondLine = wx.StaticLine(self.panel, )
        self.goButton = GenButton(self.panel, label="Start")
        color = (0, 255, 0)
        self.goButton.SetBackgroundColour(color)
        self.goButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")
        self.goButton.Bind(wx.EVT_BUTTON, self.onGo)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)
        self.updateSizer()
        self.Show()

    def onStartLiveVideo(self, event):
        self.nmesures = 0
        self.g = {'1':0.0,
                  '2':0.0,
                  '3':0.0,
                  '4':0.0,
                  '5':0.0,
                  '6':0.0,
                  '7':0.0,
                  '8':0.0}
        self.mesures = {'llarg_ull_esq':0.0,
                        'llarg_ull_dret':0.0,
                        'ample_boca':0.0,
                        'llarg_nas':0.0,
                        'ample_nas':0.0,
                        'llarg_rostre':0.0,
                        'ample_rostre':0.0,
                        'inter_pupil':0.0}
        if self.LiveVideoStream:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()
        else:
            self.LiveVideoStream = True
            self.VideoFile = False
            self.capture = cv2.VideoCapture(0)
            self.capture.set(3, 640)
            self.capture.set(4, 480)
            ret, self.frame = self.capture.read()
            if ret:
                self.height, self.width = self.frame.shape[:2]
                self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
                self.imageCtrl.SetBitmap(self.bmp)
                self.timex = wx.Timer(self)
                self.timex.Start(1000. / 24)
                self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
            else:
                print ("Error no webcam image")
            self.goButton.Enable()
            color = (255, 0, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.BtnStartLiveVideo.SetLabel("Stop Live Video Stream")
        self.updateSizer()

    def startVideo(self, videopath=None):
        self.nmesures = 0
        self.g = {'1':0.0,
                  '2':0.0,
                  '3':0.0,
                  '4':0.0,
                  '5':0.0,
                  '6':0.0,
                  '7':0.0,
                  '8':0.0}
        self.mesures = {'llarg_ull_esq':0.0,
                        'llarg_ull_dret':0.0,
                        'ample_boca':0.0,
                        'llarg_nas':0.0,
                        'ample_nas':0.0,
                        'llarg_rostre':0.0,
                        'ample_rostre':0.0,
                        'inter_pupil':0.0}
        self.VideoFile = False
        if videopath is None: return
        self.LiveVideoStream = True
        self.BtnStartLiveVideo.SetLabel("Stop Video Stream")
        color = (255, 0, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.goButton.Enable()
        self.capture = cv2.VideoCapture(videopath)
        ret, self.frame = self.capture.read()
        if ret:
            self.VideoFile = True
            h, w = self.frame.shape[:2]
            aspectratio = float(self.VideoWidth) / float(w)
            self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
            self.imageCtrl.SetBitmap(self.bmp)
            self.timex = wx.Timer(self)
            self.timex.Start(1000. / 24)
            self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
        self.updateSizer()

    def redraw(self, event):
        ret, self.frame = self.capture.read()
        if ret:
            if self.VideoFile:
                h, w = self.frame.shape[:2]
                aspectratio = float(self.VideoWidth) / float(w)
                self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if self.pupil_detector:
                self.image, self.nmesures, self.mesures, self.g = \
                    detect_and_track_pupils(image=self.frame, yverbose=self.yverbose, nmesures=self.nmesures,
                                            mesures=self.mesures, g=self.g)
                h, w = self.image.shape[:2]
                aspectratio = float(self.VideoWidth) / float(w)
                self.image = cv2.resize(self.image, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            else:
                self.image = self.frame
            self.height, self.width = self.image.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.image)
            self.bmp.CopyFromBuffer(self.image)
            self.imageCtrl.SetBitmap(self.bmp)
            self.Refresh()
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()

    def update_image_on_dnd(self, filepath):
        self.startVideo(filepath)

    def updateSizer(self):
        topSizer = wx.BoxSizer(wx.VERTICAL)
        titleSizer = wx.BoxSizer(wx.HORIZONTAL)
        textSizer = wx.BoxSizer(wx.HORIZONTAL)
        imgSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        titleSizer.Add(self.lblTitle, 1, wx.ALL | wx.EXPAND, 5)
        textSizer.Add(self.lblDragandDrop, 1, wx.ALL | wx.EXPAND, 5)
        textSizer.Add(self.BtnStartLiveVideo, 1, wx.ALL | wx.EXPAND, 5)
        imgSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        btnSizer.Add(self.goButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)

        topSizer.Add(titleSizer, 0, wx.CENTER)
        topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(textSizer, 0, wx.CENTER)
        topSizer.Add(imgSizer, 0, wx.CENTER)
        topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)

    def onCancel(self, event):
        self.Close()

    def onGo(self, event):
        if self.pupil_detector:
            self.pupil_detector = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
        else:
            self.pupil_detector = True
            self.goButton.SetLabel("Stop")
            color = (255, 0, 0)
            self.goButton.SetBackgroundColour(color)


###########################################################################
#  Class FaceRecognizerInVideoPanel
###########################################################################

class FaceRecognizerInVideoPanel(wx.Dialog):

    def __init__(self, parent):
        global Default_Settings_File
        wx.Dialog.__init__(self, None, wx.ID_ANY, "Face Detect & Recognize in Video")
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.detect_faces = False
        self.NumberImage = 0
        self.detect_face = False
        self.LiveVideoStream = False
        self.VideoFile = False
        self.capture = None
        self.frame = None
        self.height = None
        self.width = None
        self.bmp = None
        self.timex = None
        self.detect_faces = False
        self.VideoFile = None
        self.directory = None
        self.points = None
        self.r = None
        self.roi_gray = None
        self.detect = False
        self.FaceImageName = None
        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)
        self.Pred_Freq = []
        for i in range(100):
            self.Pred_Freq.append(0.0)
        self.faces, self.labels_num, self.names, self.ann, self.NewW, self.NewH, self.Pred_Freq= prepare_faces_data(self.Data_Path,
                                                                                                     self.yverbose,
                                                                                                     self.Train_Method)
        title_text = " "
        imgText = "<span foreground='red' font-size='15pt'>Drag and Drop one Video or: </span>"
        if self.Train_Method == "LBPH":
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='250%'>Method: LBPH Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "Eigen":
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Eigen Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "Fisher":
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Fisher Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "ANN":
            title_text = "<span foreground='blue' font-size='25pt'>Method: Artificial Neural networks</span>"
        else:
            print ("Error in Train Method")
            exit()

        self.lblTitle = wx.StaticText(self.panel)
        self.lblTitle.SetLabelMarkup(title_text)
        self.firstLine = wx.StaticLine(self.panel, )
        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup(imgText)
        self.BtnStartLiveVideo = GenButton(self.panel, label="Start Live Video Stream")
        color = (0, 255, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.BtnStartLiveVideo.Bind(wx.EVT_BUTTON, self.onStartLiveVideo)

        pub.subscribe(self.update_image_on_dnd, 'dnd')
        filedroptarget = DropTarget(self, False, True)

        img = wx.Image(self.VideoWidth, self.VideoHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, id=wx.ID_ANY, bitmap=wx.Bitmap(img))
        self.imageCtrl.SetDropTarget(filedroptarget)
        self.secondLine = wx.StaticLine(self.panel, )
        self.goButton = GenButton(self.panel, label="Start")
        color = (0, 255, 0)
        self.goButton.SetBackgroundColour(color)
        self.goButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")
        self.goButton.Bind(wx.EVT_BUTTON, self.onGo)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)
        self.updateSizer()
        self.Show()

    def onStartLiveVideo(self, event):
        for i in range(100):
            self.Pred_Freq.append(0.0)
        if self.LiveVideoStream:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()
        else:
            self.LiveVideoStream = True
            self.VideoFile = False
            self.capture = cv2.VideoCapture(0)
            self.capture.set(3, 640)
            self.capture.set(4, 480)
            ret, self.frame = self.capture.read()
            if ret:
                self.height, self.width = self.frame.shape[:2]
                self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
                self.imageCtrl.SetBitmap(self.bmp)
                self.timex = wx.Timer(self)
                self.timex.Start(1000. / 24)
                self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
            else:
                print ("Error no webcam image")
            self.goButton.Enable()
            color = (255, 0, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.BtnStartLiveVideo.SetLabel("Stop Live Video Stream")
        self.updateSizer()

    def startVideo(self, videopath=None):
        for i in range(100):
            self.Pred_Freq.append(0.0)
        self.VideoFile = False
        if videopath is None: return
        self.LiveVideoStream = True
        self.BtnStartLiveVideo.SetLabel("Stop Video Stream")
        color = (255, 0, 0)
        self.BtnStartLiveVideo.SetBackgroundColour(color)
        self.goButton.Enable()
        self.capture = cv2.VideoCapture(videopath)
        ret, self.frame = self.capture.read()
        if ret:
            self.VideoFile = True
            h, w = self.frame.shape[:2]
            aspectratio = float(self.VideoWidth) / float(w)
            self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
            self.imageCtrl.SetBitmap(self.bmp)
            self.timex = wx.Timer(self)
            self.timex.Start(1000. / 24)
            self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
        self.updateSizer()

    def redraw(self, event):
        ret, self.frame = self.capture.read()
        if ret:
            if self.VideoFile:
                h, w = self.frame.shape[:2]
                aspectratio = float(self.VideoWidth) / float(w)
                self.frame = cv2.resize(self.frame, (self.VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if self.detect_faces:
                if self.dlib_detector:
                    detect_and_recognize_faces_dlib(image=self.frame, names=self.names, ann=self.ann,
                                                    yverbose=self.yverbose, MaxConfidence=self.MaxConfidence,
                                                    prediction=self.Pred_Freq)
                elif self.haar_cascade_detector:
                    detect_and_recognize_faces_opencv(image=self.frame, face_recognizer=self.face_recognizer,
                                                      names=self.names, Train_Method=self.Train_Method,
                                                      MaxConfidence=self.MaxConfidence, yverbose=self.yverbose,
                                                      NewW=self.NewW, NewH=self.NewH, prediction=self.Pred_Freq)
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)
            self.bmp.CopyFromBuffer(self.frame)
            self.imageCtrl.SetBitmap(self.bmp)
            self.Refresh()
        else:
            self.LiveVideoStream = False
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
            self.goButton.Disable()
            self.BtnStartLiveVideo.SetLabel("Start Live Video Stream")
            color = (0, 255, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.timex.Stop()
            self.capture.release()
            img = wx.Image(self.VideoWidth, self.VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
            self.updateSizer()

    def update_image_on_dnd(self, filepath):
        self.startVideo(filepath)

    def updateSizer(self):
        topSizer = wx.BoxSizer(wx.VERTICAL)
        titleSizer = wx.BoxSizer(wx.HORIZONTAL)
        textSizer = wx.BoxSizer(wx.HORIZONTAL)
        imgSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        titleSizer.Add(self.lblTitle, 1, wx.ALL | wx.EXPAND, 5)
        textSizer.Add(self.lblDragandDrop, 1, wx.ALL | wx.EXPAND, 5)
        textSizer.Add(self.BtnStartLiveVideo, 1, wx.ALL | wx.EXPAND, 5)
        imgSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        btnSizer.Add(self.goButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)

        topSizer.Add(titleSizer, 0, wx.CENTER)
        topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(textSizer, 0, wx.CENTER)
        topSizer.Add(imgSizer, 0, wx.CENTER)
        topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)

    def onCancel(self, event):
        self.Close()

    def onGo(self, event):
        if self.detect_faces:
            self.Pred_Freq = []
            self.detect_faces = False
            self.goButton.SetLabel("Start")
            color = (0, 255, 0)
            self.goButton.SetBackgroundColour(color)
        else:
            self.detect_faces = True
            self.goButton.SetLabel("Stop")
            color = (255, 0, 0)
            self.goButton.SetBackgroundColour(color)


###########################################################################
## Class FaceRecognizerPanel
###########################################################################

class FaceRecognizerPanel(wx.Dialog):

    def __init__(self, parent):
        global Default_Settings_File
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Face Recognizer")
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.detect_face = False
        self.frame = None
        self.height = None
        self.width = None
        self.bmp = None
        self.directory = None
        self.points = None
        self.r = None
        self.detect = False
        self.FaceImageName = None
        self.img = None
        self.directory = None
        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)
        self.Pred_Freq = []
        self.faces, self.labels_num, self.names, self.ann, self.NewW, self.NewH, self.Pred_Freq = prepare_faces_data(self.Data_Path,
                                                                                                     self.yverbose,
                                                                                                     self.Train_Method)
        title_text = " "
        imgText = "<span foreground='red' font-size='15pt'>Drag and Drop an Image.</span>"
        if self.Train_Method == "LBPH":
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='250%'>Method: LBPH Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "Eigen":
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Eigen Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "Fisher":
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Fisher Face Recognizer</span>"
            self.face_recognizer.train(self.faces, np.array(self.labels_num))
        elif self.Train_Method == "ANN":
            title_text = "<span foreground='blue' font-size='25pt'>Method: Artificial Neural networks</span>"
        else:
            print ("Error in Train Method")
            exit()

        self.lblTitle = wx.StaticText(self.panel)
        self.lblTitle.SetLabelMarkup(title_text)
        self.firstLine = wx.StaticLine(self.panel, )
        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup(imgText)

        pub.subscribe(self.update_image_on_dnd, 'dnd')
        filedroptarget = DropTarget(self, False)

        bmp = wx.Image(self.VideoWidth, self.VideoHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, id=wx.ID_ANY, bitmap=wx.Bitmap(bmp))
        self.imageCtrl.SetDropTarget(filedroptarget)
        self.secondLine = wx.StaticLine(self.panel, )
        self.OKButton = GenButton(self.panel, label="OK")
        self.OKButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")
        self.OKButton.Bind(wx.EVT_BUTTON, self.onOK)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)

        self.updateSizer()
        self.Show()

    def updateSizer(self):
        topSizer = wx.BoxSizer(wx.VERTICAL)
        titleSizer = wx.BoxSizer(wx.HORIZONTAL)
        textSizer = wx.BoxSizer(wx.HORIZONTAL)
        imgSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        titleSizer.Add(self.lblTitle, 1, wx.ALL | wx.EXPAND, 5)
        textSizer.Add(self.lblDragandDrop, 1, wx.ALL | wx.EXPAND, 5)
        imgSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        btnSizer.Add(self.OKButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)

        topSizer.Add(titleSizer, 0, wx.CENTER)
        topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(textSizer, 0, wx.CENTER)
        topSizer.Add(imgSizer, 0, wx.CENTER)
        topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)
        self.panel.Update()
        self.panel.Refresh()

    def onCancel(self, event):
        self.Close()

    def onOK(self, event):
        pass

    def update_image_on_dnd(self, filepath):
        self.img = cv2.imread(filepath)
        if self.img is None: return
        self.img = scale_image(self.img, self.VideoWidth, self.VideoHeight)
        if self.dlib_detector:
            detect_and_recognize_faces_dlib(image=self.img, names=self.names, ann=self.ann,
                                            yverbose=self.yverbose, MaxConfidence=self.MaxConfidence,
                                            prediction=self.Pred_Freq)
        elif self.haar_cascade_detector:
            detect_and_recognize_faces_opencv(image=self.img, face_recognizer=self.face_recognizer, names=self.names,
                                              Train_Method=self.Train_Method, MaxConfidence=self.MaxConfidence,
                                              NewW=self.NewW, NewH=self.NewH, prediction=self.Pred_Freq)
        self.on_view()
        self.OKButton.Enable()

    def on_view(self):
        self.height, self.width = self.img.shape[:2]
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, rgb)
        self.bmp.CopyFromBuffer(rgb)
        self.imageCtrl.SetBitmap(self.bmp)
        self.imageCtrl.Center(wx.BOTH)
        self.updateSizer()


###########################################################################
## Class NewFacePanel
###########################################################################

class NewFacePanel(wx.Dialog):

    def __init__(self, parent):
        global Default_Settings_File
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "New Face Input")
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.FaceImageName = None
        self.InfoFaceName = None
        self.points = None
        self.r = None
        self.img = None
        self.roi_gray = None
        self.detect = False
        self.width = None
        self.height = None
        self.bmp = None
        self.directory = None

        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)

        self.DataPath = self.Data_Path
        self.NumberImage = 0
        self.detect_face = False

        self.lblFaceName = wx.StaticText(self.panel, label="Face Name: ")
        self.lblErrorFaceName = wx.StaticText(self.panel)
        self.lblErrorFaceName.SetLabelMarkup(
            "<span foreground='red'>Attention: the name exists in Database. Try again.</span>")
        self.lblErrorFaceName.Hide()
        self.FaceName = wx.TextCtrl(self.panel, value="", size=(300, -1))
        self.BtnFaceName = GenButton(self.panel, label="Create")
        self.BtnFaceName.Bind(wx.EVT_BUTTON, self.onFaceName)

        self.firstLine = wx.StaticLine(self.panel, )

        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup("<span foreground='red'>Drag and Drop an Image.</span>")
        self.lblDragandDrop.Hide()
        pub.subscribe(self.update_image_on_dnd, 'dnd')
        bmp = wx.Image(self.VideoWidth, self.VideoHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, id=wx.ID_ANY, bitmap=wx.Bitmap(bmp), style=0)
        self.imageCtrl.Hide()
        filedroptarget = DropTarget(self, True)
        self.imageCtrl.SetDropTarget(filedroptarget)

        self.secondLine = wx.StaticLine(self.panel, )

        self.saveButton = GenButton(self.panel, label="Save")
        self.saveButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)

        self.updateSizer()
        self.Show()

    def updateSizer(self):
        topSizer = wx.BoxSizer(wx.VERTICAL)
        faceNameSizer = wx.BoxSizer(wx.HORIZONTAL)
        textErrorSizer = wx.BoxSizer(wx.HORIZONTAL)
        textImageSizer = wx.BoxSizer(wx.HORIZONTAL)
        ImageSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        faceNameSizer.Add(self.lblFaceName, 0, wx.ALL, 5)
        faceNameSizer.Add(self.FaceName, 1, wx.ALL | wx.EXPAND, 5)
        faceNameSizer.Add(self.BtnFaceName, 0, wx.ALL, 5)
        textErrorSizer.Add(self.lblErrorFaceName, 0, wx.ALL, 5)
        textImageSizer.Add(self.lblDragandDrop, 0, wx.ALL, 5)
        ImageSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        btnSizer.Add(self.saveButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)

        topSizer.Add(faceNameSizer, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(textImageSizer, 0, wx.CENTER)
        topSizer.Add(textErrorSizer, 0, wx.CENTER)
        topSizer.Add(ImageSizer, 0, wx.CENTER)
        topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)
        self.Refresh()

    def onFaceName(self, event):
        self.Name = str(self.FaceName.GetValue())
        if self.Name == "": return
        dirs = os.listdir(self.Data_Path)
        dirs2 = dirs
        for item in dirs:
            if item.startswith("."):
                dirs2.remove(item)
        dirs = dirs2
        for item in dirs:
            if item == self.Name:
                self.lblErrorFaceName.Show()
                self.updateSizer()
                return
        self.directory = self.Data_Path + "/" + self.Name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.lblErrorFaceName.Hide()
            self.lblDragandDrop.Show()
            self.imageCtrl.Show()
            self.FaceName.Disable()
            self.BtnFaceName.Disable()
        self.updateSizer()

    def onCancel(self, event):
        self.Close()

    def onSave(self, event):
        self.FaceImageName = self.directory + "/" + str(self.NumberImage) + ".png"
        self.InfoFaceName = self.directory + "/" + str(self.NumberImage) + ".info"
        filename = open(self.InfoFaceName, "w+")
        for item in self.r:
            filename.write("%s\n" % item)
        filename.close()
        cv2.imwrite(self.FaceImageName, self.roi_gray)
        self.NumberImage = self.NumberImage + 1
        self.saveButton.Disable()

    def update_image_on_dnd(self, filepath):
        self.img = cv2.imread(filepath)
        if self.img is None: return
        self.img = scale_image(self.img, self.VideoWidth, self.VideoHeight)
        self.img, self.roi_gray, self.detect, self.points, self.r = detect_face_dlib(image=self.img)
        #
        # Draw the face landmarks on the screen.
        #
        if self.detect and self.yverbose:
           color = (255, 0, 0)
           for (x, y) in self.points:
               cv2.circle(self.img, (x, y), 1, color, -1)
        self.on_view()
        if self.detect:
            self.saveButton.Enable()

    def on_view(self):
        self.height, self.width = self.img.shape[:2]
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, rgb)
        self.bmp.CopyFromBuffer(rgb)
        self.imageCtrl.SetBitmap(self.bmp)
        self.imageCtrl.Center(wx.BOTH)
        self.panel.Refresh()
        self.updateSizer()


###########################################################################
## Class SettingsPanel
###########################################################################

class SettingsPanel(wx.Dialog):

    def __init__(self, parent):
        global Default_Settings_File
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Settings Input")
        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)
        self.LoadSettings = False
        self.width = ['1024', '800', '640', '480']
        self.lblList = ["16:9", "4:3"]
        self.methods = ['LBPH', 'Eigen', 'Fisher', 'ANN']
        self.verbose = ['Silent', 'Output All']
        self.Text1 = "Video Size: " + str(self.VideoWidth) + "x" + str(self.VideoHeight)
        self.Tverbose = "Output All"
        self.FileName = None
        if self.yverbose:
            self.Tverbose = "Output All"
        else:
            self.Tverbose = "Silent"
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.topSizer = wx.BoxSizer(wx.VERTICAL)

        self.lblSettingsFile = wx.StaticText(self.panel, label="Settings file: ")
        self.SettingsFile = wx.TextCtrl(self.panel, value="Default_Settings.config")
        self.SettingsFile.Disable()
        self.BtnSettingsFile = GenButton(self.panel, label="Load")
        self.FileName = str(self.SettingsFile.GetValue())
        self.lblPhotosPath = wx.StaticText(self.panel, label="Data Path:     ")
        self.DataPath = wx.TextCtrl(self.panel, value=self.Data_Path)
        self.DataPathText = str(self.DataPath.GetValue())
        self.DataPath.Disable()
        self.BtnDataPath = GenButton(self.panel, label="Load")
        self.firstLine = wx.StaticLine(self.panel, )
        self.cmbText = wx.StaticText(self.panel, -1, "Select Aspect Ratio: ")
        self.rbAspectRatio = wx.RadioBox(self.panel, label='', choices=self.lblList, majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)
        self.rbAspectRatio.SetStringSelection(self.AspectRatio)
        self.Aspect = str(self.rbAspectRatio.GetStringSelection())
        self.selectWidthText = wx.StaticText(self.panel, -1, "Select Width: ")
        self.cmbWidth = wx.ComboBox(self.panel, -1, "800", choices=self.width)
        self.cmbWidth.SetStringSelection(str(self.VideoWidth))
        self.sizeText1 = wx.StaticText(self.panel, -1, self.Text1)
        self.secondLine = wx.StaticLine(self.panel, )
        self.selectMethodText = wx.StaticText(self.panel, -1, "Select Method: ")
        self.cmbMethod = wx.ComboBox(self.panel, -1, "LBPH", choices=self.methods)
        self.cmbMethod.SetStringSelection(self.Train_Method)
        self.lblMaxConfidence = wx.StaticText(self.panel, label="Max Confidence: ")
        self.txtMaxConfidence = wx.TextCtrl(self.panel, value="75.0")
        self.txtMaxConfidence.SetLabel(str(self.MaxConfidence))
        self.selectVerboseText = wx.StaticText(self.panel, -1, "Output option: ")
        self.cmbVerbose = wx.ComboBox(self.panel, -1, "Silent", choices=self.verbose)
        self.cmbVerbose.SetStringSelection(self.Tverbose)
        self.thirdLine = wx.StaticLine(self.panel, )
        self.saveButton = GenButton(self.panel, label="Save")
        self.saveButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")

        self.BtnSettingsFile.Bind(wx.EVT_BUTTON, self.onSettingsFile)
        self.BtnDataPath.Bind(wx.EVT_BUTTON, self.onDataPath)
        self.rbAspectRatio.Bind(wx.EVT_RADIOBOX, self.onAspectRatio)
        self.cmbWidth.Bind(wx.EVT_COMBOBOX, self.oncmbWidth)
        self.cmbMethod.Bind(wx.EVT_COMBOBOX, self.oncmbMethod)
        self.cmbVerbose.Bind(wx.EVT_COMBOBOX, self.oncmbVerbose)
        self.txtMaxConfidence.Bind(wx.EVT_TEXT, self.onMaxConfidence)
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)

        self.getVideoDimensions()
        self.UpdateSizer()

        self.Show()

    def UpdateSizer(self):
        self.topSizer = wx.BoxSizer(wx.VERTICAL)
        settingsFileSizer = wx.BoxSizer(wx.HORIZONTAL)
        pathSizer = wx.BoxSizer(wx.HORIZONTAL)
        selectwidthSizer = wx.BoxSizer(wx.HORIZONTAL)
        selectmethodSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        settingsFileSizer.Add(self.lblSettingsFile, 0, wx.ALL, 5)
        settingsFileSizer.Add(self.SettingsFile, 1, wx.ALL | wx.EXPAND, 5)
        settingsFileSizer.Add(self.BtnSettingsFile, 0, wx.ALL, 5)
        pathSizer.Add(self.lblPhotosPath, 0, wx.ALL, 5)
        pathSizer.Add(self.DataPath, 1, wx.ALL | wx.EXPAND, 5)
        pathSizer.Add(self.BtnDataPath, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.cmbText, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.rbAspectRatio, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.selectWidthText, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.cmbWidth, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.sizeText1, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.selectMethodText, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.cmbMethod, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.lblMaxConfidence, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.txtMaxConfidence, 1, wx.ALL | wx.EXPAND, 5)
        selectmethodSizer.Add(self.selectVerboseText, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.cmbVerbose, 0, wx.ALL, 5)
        btnSizer.Add(self.saveButton, 0, wx.ALL, 5)
        btnSizer.Add(self.closeButton, 0, wx.ALL, 5)
        self.topSizer.Add(settingsFileSizer, 0, wx.CENTER | wx.EXPAND)
        self.topSizer.Add(pathSizer, 0, wx.CENTER | wx.EXPAND)
        self.topSizer.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 5)
        self.topSizer.Add(selectwidthSizer, 0, wx.CENTER | wx.EXPAND)
        self.topSizer.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 5)
        self.topSizer.Add(selectmethodSizer, 0, wx.CENTER | wx.EXPAND)
        self.topSizer.Add(self.thirdLine, 0, wx.ALL | wx.EXPAND, 5)
        self.topSizer.Add(btnSizer, 0, wx.ALL | wx.CENTER, 5)

        self.panel.SetSizer(self.topSizer)
        self.topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)

    def getVideoDimensions(self):
        self.Aspect = str(self.rbAspectRatio.GetStringSelection())
        heightA = ['576', '450', '360', '270']
        heightB = ['768', '600', '480', '360']
        if self.rbAspectRatio.GetStringSelection() == "16:9":
            self.VideoWidth = self.width[self.width.index(self.cmbWidth.GetValue())]
            self.VideoHeight = heightA[self.width.index(self.cmbWidth.GetValue())]
        elif self.rbAspectRatio.GetStringSelection() == "4:3":
            self.VideoWidth = self.width[self.width.index(self.cmbWidth.GetValue())]
            self.VideoHeight = heightB[self.width.index(self.cmbWidth.GetValue())]
        self.Text1 = "Video Size: " + str(self.VideoWidth) + "x" + str(self.VideoHeight)
        self.sizeText1.SetLabel(self.Text1)
        self.sizeText1.Update()
        self.sizeText1.Refresh()
        self.AspectRatio = self.rbAspectRatio.GetStringSelection()
        self.VideoWidth = int(self.VideoWidth)
        self.VideoHeight = int(self.VideoHeight)
        self.UpdateSizer()
        return

    def onCancel(self, event):
        self.Close()

    def onSave(self, event):
        writeSettings(self.FileName, self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight,
                      self.Train_Method, self.MaxConfidence, self.yverbose)
        self.saveButton.Disable()

    def onMaxConfidence(self, event):
        if self.LoadSettings: return
        self.MaxConfidence = float(self.txtMaxConfidence.GetValue())
        self.saveButton.Enable()

    def oncmbMethod(self, event):
        if self.LoadSettings: return
        self.Train_Method = self.cmbMethod.GetStringSelection()
        self.saveButton.Enable()

    def oncmbVerbose(self, event):
        if self.LoadSettings: return
        self.Tverbose = self.cmbVerbose.GetStringSelection()
        if self.Tverbose == "Output All":
            self.yverbose = True
        elif self.Tverbose == "Silent":
            self.yverbose = False
        self.saveButton.Enable()

    def oncmbWidth(self, event):
        if self.LoadSettings: return
        self.getVideoDimensions()
        self.saveButton.Enable()

    def onAspectRatio(self, event):
        if self.LoadSettings: return
        self.getVideoDimensions()
        self.saveButton.Enable()

    def onSettingsFile(self, event):
        global Default_Settings_File
        if self.LoadSettings: return
        openFileDialog = wx.FileDialog(self, "Open", "", "", "Settings files (*.config)|*.config",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.SetDirectory("./.Config/")
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return
        self.FileName = openFileDialog.GetPath()
        self.SettingsFile.SetValue(self.FileName)
        self.LoadSettings = False
        self.saveButton.Enable()
        if os.path.isfile(self.FileName):
            self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
            self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
                self.FileName)
            self.DataPathText = self.Data_Path
            self.Aspect = self.AspectRatio
            self.DataPath.SetValue(self.DataPathText)
            self.rbAspectRatio.SetStringSelection(self.Aspect)
            self.cmbWidth.SetStringSelection(str(self.VideoWidth))
            self.cmbMethod.SetStringSelection(self.Train_Method)
            self.txtMaxConfidence.SetValue(str(self.MaxConfidence))
            if self.yverbose:
                Tverbose = "Output All"
            else:
                Tverbose = "Silent"
            self.saveButton.Disable()
            self.cmbVerbose.SetStringSelection(Tverbose)
            self.getVideoDimensions()
            self.UpdateSizer()
            self.Refresh()
        Default_Settings_File = self.FileName
        openFileDialog.Destroy()

    def onDataPath(self, event):
        if self.LoadSettings: return
        self.LoadSettings = False
        openDirDialog = wx.DirDialog(self, "Choose Data directory", str(self.DataPath.GetValue()),
                                     wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        openDirDialog.SetPath("./")
        if openDirDialog.ShowModal() == wx.ID_CANCEL:
            return
        self.DataPathText = str(self.DataPath.GetValue())
        self.DataPath.SetValue(openDirDialog.GetPath())
        self.saveButton.Enable()
        self.UpdateSizer()
        openDirDialog.Destroy()


###########################################################################
## Class MainWindow
###########################################################################

# noinspection PyArgumentList
class MainWindow(wx.Frame):

    def __init__(self, parent, *args, **kwargs):
        global Default_Settings_File
        super(MainWindow, self).__init__(*args, **kwargs)
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="Face Recognizer", pos=wx.DefaultPosition,
                          size=(600, 200), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.panel = wx.Panel(self)
        self.parent = parent
        self.Data_Path, self.AspectRatio, self.VideoWidth, self.VideoHeight, self.Train_Method, \
        self.haar_cascade_detector, self.dlib_detector, self.MaxConfidence, self.yverbose = loadSettings(
            Default_Settings_File)
        self.Exit = None
        self.Configure = None
        self.New_Face = None
        self.New_Face_from_Video = None
        self.Recognize_Face = None
        self.Recognize_Face_in_Video = None
        self.FileName = None
        self.Pupil_Tracking_in_Video = None
        self.InitUI()

    def InitUI(self):
        # self.SetSizeHints(wx.DefaultSize,wx.DefaultSize)
        SizerVertical = wx.BoxSizer(wx.VERTICAL)
        SizerHorizont = wx.BoxSizer(wx.HORIZONTAL)

        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()
        AddFacesMenu = wx.Menu()
        RecognizeFacesMenu = wx.Menu()
        configureMenuItem = fileMenu.Append(wx.NewId(), "Configure", "Change the Settings")
        exitMenuItem = fileMenu.Append(wx.NewId(), "Exit", "Exit the application")
        addFaces = AddFacesMenu.Append(wx.NewId(), "Add New face", "Add New Face from Images")
        addFacesFromVideo = AddFacesMenu.Append(wx.NewId(), "Add New face from Video", "Add New Face from Video")
        recognizeFaces = RecognizeFacesMenu.Append(wx.NewId(), "Recognize faces", "Recognize Face in Images")
        recognizeFacesinVideo = RecognizeFacesMenu.Append(wx.NewId(), "Recognize faces in video",
                                                          "Recognize Faces in Video")
        pupilsTrackinginVideo = RecognizeFacesMenu.Append(wx.NewId(), "Pupils tracking in video",
                                                          "Pupils tracking in Video")
        self.Bind(wx.EVT_MENU, self.onQuit, exitMenuItem)
        self.Bind(wx.EVT_MENU, self.onSettings, configureMenuItem)
        self.Bind(wx.EVT_MENU, self.onNewFace, addFaces)
        self.Bind(wx.EVT_MENU, self.onNewFaceFromVideo, addFacesFromVideo)
        self.Bind(wx.EVT_MENU, self.onRecognizeFace, recognizeFaces)
        self.Bind(wx.EVT_MENU, self.onRecognizeFaceInVideo, recognizeFacesinVideo)
        self.Bind(wx.EVT_MENU, self.onPupilsTrackinginVideo, pupilsTrackinginVideo)
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(AddFacesMenu, "&Add Faces")
        menuBar.Append(RecognizeFacesMenu, "&Recognize Faces")
        self.SetMenuBar(menuBar)

        self.ToolBar = wx.ToolBar(self, wx.ID_ANY)
        self.ToolBar.SetToolBitmapSize((50, 50))
        self.Exit = self.ToolBar.AddTool(wx.ID_ANY, u"Exit", wx.Bitmap(u"./.icons/icons8-exit.png"))
        self.Configure = self.ToolBar.AddTool(wx.ID_ANY, u"Configure", wx.Bitmap(u"./.icons/icons8-settings.png"))
        self.ToolBar.AddSeparator()
        self.New_Face = self.ToolBar.AddTool(wx.ID_ANY, u"New Face", wx.Bitmap(u"./.icons/icons8-add_face.png"))
        self.New_Face_from_Video = self.ToolBar.AddTool(wx.ID_ANY, u"New Face from Video",
                                                        wx.Bitmap(u"./.icons/icons8-add_face_from_video.png"))
        self.ToolBar.AddSeparator()
        self.Recognize_Face = self.ToolBar.AddTool(wx.ID_ANY, u"Recognize Face",
                                                   wx.Bitmap(u"./.icons/icons8-recognize_face.png"))
        self.Recognize_Face_in_Video = self.ToolBar.AddTool(wx.ID_ANY, u"Recognize Face from Video",
                                                            wx.Bitmap(u"./.icons/icons8-recognize_in_video.png"))
        self.ToolBar.AddSeparator()
        self.Pupil_Tracking_in_Video = self.ToolBar.AddTool(wx.ID_ANY, u"Pupil Tracking in Video",
                                                            wx.Bitmap(u"./.icons/icons8-pupil_tracking.png"))
        self.ToolBar.Realize()

        self.Bind(wx.EVT_TOOL, self.onQuit, self.Exit)
        self.Bind(wx.EVT_TOOL, self.onSettings, self.Configure)
        self.Bind(wx.EVT_TOOL, self.onNewFace, self.New_Face)
        self.Bind(wx.EVT_TOOL, self.onNewFaceFromVideo, self.New_Face_from_Video)
        self.Bind(wx.EVT_TOOL, self.onRecognizeFace, self.Recognize_Face)
        self.Bind(wx.EVT_TOOL, self.onRecognizeFaceInVideo, self.Recognize_Face_in_Video)
        self.Bind(wx.EVT_TOOL, self.onPupilsTrackinginVideo, self.Pupil_Tracking_in_Video)

        SizerVertical.Add(self.ToolBar, 0, wx.EXPAND, 5)
        SizerHorizont.Add(self.ToolBar, 0, wx.EXPAND, 5)

        self.Layout()
        self.Center(wx.BOTH)

    def onPupilsTrackinginVideo(self, event):
        dlg_Pupil_Tracking = PupilTrackingInVideoPanel(parent=self.panel)
        dlg_Pupil_Tracking.ShowModal()
        dlg_Pupil_Tracking.Destroy()

    def onNewFaceFromVideo(self, event):
        dlg_NewFaceFromVideo = NewFaceFromVideoPanel(parent=self.panel)
        dlg_NewFaceFromVideo.ShowModal()
        dlg_NewFaceFromVideo.Destroy()

    def onRecognizeFace(self, event):
        dlg_FaceRecognizer = FaceRecognizerPanel(parent=self.panel)
        dlg_FaceRecognizer.ShowModal()
        dlg_FaceRecognizer.Destroy()
        self.Enable(True)

    def onRecognizeFaceInVideo(self, event):
        dlg_FaceRecognizer = FaceRecognizerInVideoPanel(parent=self.panel)
        dlg_FaceRecognizer.ShowModal()
        dlg_FaceRecognizer.Destroy()
        self.Enable(True)

    def onQuit(self, event):
        self.Close()

    def onSettings(self, event):
        dlg_Settings = SettingsPanel(parent=self.panel)
        dlg_Settings.ShowModal()
        dlg_Settings.Destroy()
        self.Enable(True)

    def onNewFace(self, event):
        dlg_NewFace = NewFacePanel(parent=self.panel)
        dlg_NewFace.ShowModal()
        dlg_NewFace.Destroy()
        self.Enable(True)

    def __del__(self):
        pass


def main():
    app = wx.App()
    window = MainWindow(None)
    window.Centre(wx.BOTH)
    window.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
