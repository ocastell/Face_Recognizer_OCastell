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
import math
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
from wx.lib.buttons import GenButton
from wx.lib.pubsub import pub

#
# Initial Values (Global Variables)
#
global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
MaxConfidence = 75.
yverbose = True
Train_Method = "LBPH"
Data_Path = "./Data"
AspectRatio = "16:9"
VideoWidth = 800
VideoHeight = 450
MaxTakeImages = 20
(im_width, im_height) = (240, 240)
Default_Settings_File = "./Default_Settings.config"
#
# OpenCV files haar_cascade
#
haar_cascade_file = './openCV/' + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_file)
haar_cascade_detector = False
#
# Create a HOG face detector using the built-in dlib class
#
predictor_model = "./dlib/" + "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
dlib_detector = False

colors = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))


def writeSettings(filename):
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
    file = open(filename, "w+")
    file.write(Data_Path + "\n")
    file.write(AspectRatio + "\n")
    file.write(str(VideoWidth) + "\n")
    file.write(str(VideoHeight) + "\n")
    file.write(Train_Method + "\n")
    file.write(str(MaxConfidence) + "\n")
    if yverbose:
        Tverbose = "Output All"
    else:
        Tverbose = "Silent"
    file.write(Tverbose)
    file.close()


def loadSettings():
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
    file = open(Default_Settings_File, "r")
    num_lin = 1
    for line in file:
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
    file.close()


def detect_and_recognize_faces(image=None, face_recognizer=None, names=None, ann=None):
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
    if Train_Method == "ANN":
        dlib_detector = True
        haar_cascade_detector = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if haar_cascade_detector:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) == 0):
            return None
        ncol = 0
        for (x, y, w, h) in faces:
            img_test = gray[y:y + h, x:x + w]
            if Train_Method <> "LBPH":
                img_test = cv2.resize(img_test, (im_width, im_height), cv2.INTER_LANCZOS4)
            label, confidence = face_recognizer.predict(img_test)
            color = colors[ncol]
            draw_rectangle(image, (x, y, w, h), color)
            if confidence <= MaxConfidence:
                text = names[int(label)]
            else:
                text = "Unknown"
            draw_text(image, text, x, y, w, h, color)
            if yverbose:
                print "This person is " + text + " with a confidence of " + str(confidence)
            ncol = ncol + 1
        return
    elif dlib_detector:
        face = face_detector(image, 1)
        k = 0
        ncol = 0
        for i, face_rect in enumerate(face):
            color = colors[ncol]
            (x, y, w, h) = dlib_to_cv2_rectangle(face_rect)
            #
            # Get the the face's landmarks
            #
            pose_landmarks = face_pose_predictor(image, face_rect)
            points = shape_to_np(pose_landmarks)
            r = get_face_metric(points, pupil_distance=0.0)
            test_matrix = [r]
            test_matrix = np.array(test_matrix, np.float32)
            _re, pred = ann.predict(test_matrix)
            indexMax = np.argmax(pred)
            norm = 0
            for k in pred[0]:
                norm = norm + float(k) ** 2
            norm = math.sqrt(norm)
            normed = []
            for i in pred[0]:
                normed.append(float(i) / norm)
            confidence = normed[np.argmax(normed)]
            text = names[indexMax] + " " + "%0.2f" % confidence
            draw_rectangle(image, (x, y, w, h), color)
            draw_text(image, text, x, y, w, h, color)
            if yverbose:
                print "This person is " + text + " with a confidence of " + str(confidence)
                print pred
            ncol = ncol + 1
        return
    return


def getdistance(point_A=None, point_B=None):
    a = np.array(point_A)
    b = np.array(point_B)
    distance = np.linalg.norm(a - b)
    return distance


def getmiddlepoint(point_A=None, point_B=None):
    a = np.array(point_A)
    b = np.array(point_B)
    middlepoint = (a + b) / 2
    return (middlepoint[0], middlepoint[1])


def get_face_metric(points, pupil_distance=0.0):
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
    conversion = 1.0
    if pupil_distance <> 0:
        conversion = 64.0 / pupil_distance
        interpupil_distance = int(pupil_distance)
    interocular_distance = int(abs(getdistance(points[39], points[42])))
    left_eye_width = int(abs(getdistance(points[45], points[42])))
    left_eye_height = int(
        abs(getdistance(getmiddlepoint(points[43], points[44]), getmiddlepoint(points[47], points[46]))))
    right_eye_width = int(abs(getdistance(points[36], points[39])))
    right_eye_height = int(
        abs(getdistance(getmiddlepoint(points[37], points[38]), getmiddlepoint(points[41], points[40]))))
    nose_height = int(abs(getdistance(points[33], points[27])))
    nose_width = int(abs(getdistance(points[31], points[35])))
    upper_lip_height = int(abs(getdistance(points[51], points[62])))
    lower_lip_height = int(abs(getdistance(points[57], points[67])))
    lip_height = int(abs(getdistance(points[51], points[57])))
    facial_width = int(abs(getdistance(points[16], points[0])))
    upper_facial_height = int(abs(getdistance(points[33], points[27])))
    lower_facial_height = int(abs(getdistance(points[33], points[8])))
    facial_height = abs(getdistance(points[27], points[8]))
    jaw_width = int(abs(getdistance(points[11], points[5])))
    middle_point = getmiddlepoint(points[11], points[5])
    # lower_facial_height = int(getdistance(points[8], middle_point))
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
    # Extract the Charateristic feauters of the face
    #
    r = []
    r.append(float(nose_width) / float(nose_height))
    r.append(float(left_eye_width) / float(left_eye_height))
    r.append(float(right_eye_width) / float(right_eye_height))
    r.append(float(getdistance(points[27], points[0])) / float(nose_height))
    r.append(float(getdistance(points[27], points[16])) / float(nose_height))
    r.append(float(getdistance(points[27], points[0])) / float(getdistance(points[31], points[36])))
    r.append(float(getdistance(points[27], points[16])) / float(getdistance(points[35], points[45])))
    r.append(float(getdistance(points[31], points[36])) / float(getdistance(points[31], points[27])))
    r.append(float(getdistance(points[45], points[35])) / float(getdistance(points[35], points[27])))
    r.append(fwhr)
    #
    # Print Info
    #
    # print " Interocular distance (pixels, mm): ", interocular_distance, int(interocular_distance * conversion)
    ##print " Interpupil distance (pixels, mm): ", interpupil_distance, int(interpupil_distance * conversion)
    # print " Left eye width (pixels, mm): ", left_eye_width, int(left_eye_width * conversion)
    # print " Left eye height (pixels, mm): ", left_eye_height, int(left_eye_height * conversion)
    # print " Right eye width (pixels, mm): ", right_eye_width, int(right_eye_width * conversion)
    # print " Right eye height (pixels, mm): ", right_eye_height, int(right_eye_height * conversion)
    # print " Nose height (pixels, mm): ", nose_height, int(nose_height * conversion)
    # print " Nose width (pixels, mm): ", nose_width, int(nose_width * conversion)
    # print " Upper lip height (pixels, mm): ", upper_lip_height, int(upper_lip_height * conversion)
    # print " Lower lip height (pixels, mm): ", lower_lip_height, int(lower_lip_height * conversion)
    # print " Lip height (pixels, mm): ", lip_height, int(lip_height * conversion)
    # print " Facial width (pixels, mm): ", facial_width, int(facial_width * conversion)
    # print " Upper Facial height (pixels, mm): ", upper_facial_height, int(upper_facial_height * conversion)
    # print " Lower Facial height (pixels, mm): ", lower_facial_height, int(lower_facial_height * conversion)
    # print " Facial height (pixels, mm): ", int(facial_height), int(facial_height * conversion)
    # print " Jaw width (pixels, mm): ", jaw_width, int(jaw_width * conversion)
    # print " Facial-Width-Height ratio is: ", fwhr
    return r


def dlib_to_cv2_rectangle(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.right() - x
    h = face_rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def detect_face(image=None, detect=False):
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence, r
    roi_gray = None
    points = None
    dlib_detector = None
    haar_cascade_detector = None
    (x, y, w, h) = (None, None, None, None)
    if Train_Method == "ANN":
        dlib_detector = True
        haar_cascade_detector = False
    detect = False
    color = (0, 0, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if haar_cascade_detector:
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(face) == 0):
            detect = False
            return image, gray, detect
        detect = True
        (x, y, w, h) = face[0]
    elif dlib_detector:
        face = face_detector(image, 1)
        k = 0
        for i, face_rect in enumerate(face):
            (x, y, w, h) = dlib_to_cv2_rectangle(face_rect)
            detect = True
            #
            # Get the the face's landmarks
            #
            pose_landmarks = face_pose_predictor(image, face_rect)
            points = shape_to_np(pose_landmarks)
            r = get_face_metric(points, pupil_distance=0.0)
            break
    if detect:
        draw_rectangle(image, (x, y, w, h), color)
        roi_gray = gray[y:y + h, x:x + w]
        aspectratio = float(h) / float(w)
        roi_gray = cv2.resize(roi_gray, (im_width, int(im_height * aspectratio)), cv2.INTER_LANCZOS4)
    return image, roi_gray, detect, points


def draw_rectangle(img, rect, color=(0, 255, 0)):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def draw_text(img, text, x, y, w, h, color=(0, 255, 0)):
    cv2.putText(img, text, (x + 5, y + h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2)


def prepare_faces_data():
    global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
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
                file = open(image_path, "r")
                characteristics_list = []
                for line in file:
                    line = line.strip()
                    characteristics_list.append(float(line))
                samples.append(characteristics_list)
                nsamples = nsamples + 1
            elif file_extension == ".png":
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                labels.append(dir_name)
                faces.append(image)
            if yverbose:
                if file_extension == ".info":
                    print "Name of data file : ", dir_name, image_path
                    print "Adding image charcteristics to inputs : "
                elif file_extension == ".png":
                    print "Name, src: ", dir_name, image_path
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

    # targets = -1 * np.ones((len(samples), noutput_layer), dtype=np.float32)
    targets = np.zeros((len(samples), len(names)), dtype=np.float32)
    ninput_layer = len(samples[0])
    nhidden_layer = int(1.5 * ninput_layer)
    noutput_layer = len(targets[0])
    layers = np.array([ninput_layer, nhidden_layer, noutput_layer], dtype=np.uint8)
    inputs = np.empty((noutput_layer, ninput_layer), dtype=np.float32)
    inputs_test = np.random.rand(noutput_layer, ninput_layer)
    k = 0
    clase = 0
    for max in num_samples:
        for j in range(max):
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

    if yverbose:
        print "Number of faces: ", len(names)
        print "Number of images: ", len(labels_num)
        counter = collections.Counter(labels_num)
        for item in counter.keys():
            print "Number of images for face " + names[int(item)] + " is " + str(counter.values()[int(item)])
        print "ANN - Number of input layers :  ", ninput_layer
        print "ANN - Number of hidden layers: ", nhidden_layer
        print "ANN - Number of output layers: ", noutput_layer
        print "ANN - Model saved in file    : ", file_model_name
        print "ANN - Test of model          : ", file_model_name
        for i in range(len(labels_num) - 1):
            matrix_test = [samples[i, :]]
            test = np.array(matrix_test, np.float32)
            _ret, resp = ann.predict(test)
            indexMax = np.argmax(resp)
            print "         Vector test " + str(i) + " is : " + names[indexMax]
    return faces, labels_num, names, ann


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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Add Face from Video")
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.DataPath = Data_Path
        self.NumberImage = 0
        self.detect_face = False
        self.LiveVideoStream = False

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
        img = wx.Image(VideoWidth, VideoHeight)
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
            img = wx.Image(VideoWidth, VideoHeight)
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
                img = wx.Image(VideoWidth, VideoHeight)
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
        if filepath == None: return
        self.capture = cv2.VideoCapture(filepath)
        ret, self.frame = self.capture.read()
        if ret:
            self.VideoFile = True
            h, w = self.frame.shape[:2]
            aspectratio = float(VideoWidth) / float(w)
            self.frame = cv2.resize(self.frame, (VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
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
            img = wx.Image(VideoWidth, VideoHeight)
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
        dirs = os.listdir(Data_Path)
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
        self.directory = Data_Path + "/" + self.Name
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        ret, self.frame = self.capture.read()
        if ret:
            if self.VideoFile:
                h, w = self.frame.shape[:2]
                aspectratio = float(VideoWidth) / float(w)
                self.frame = cv2.resize(self.frame, (VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.detect = False
            if self.detect_face:
                self.frame, self.roi_gray, self.detect, self.points = detect_face(image=self.frame)
                if dlib_detector:
                    #
                    # Draw the face landmarks on the screen.
                    #
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
            img = wx.Image(VideoWidth, VideoHeight)
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence, r
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
        file = open(self.InfoFaceName, "w+")
        for item in r:
            file.write("%s\n" % item)
        file.close()
        print " photo ", self.FaceImageName
        cv2.imwrite(self.FaceImageName, self.roi_gray)
        self.NumberImage = self.NumberImage + 1


###########################################################################
#  Class FaceRecognizerInVideoPanel
###########################################################################

class FaceRecognizerInVideoPanel(wx.Dialog):

    def __init__(self, parent):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Face Detect & Recognize in Video")
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.detect_faces = False
        self.LiveVideoStream = False

        self.faces, self.labels_num, self.names, self.ann = prepare_faces_data()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        title_text = " "
        imgText = "<span foreground='red' font-size='15pt'>Drag and Drop one Video or: </span>"
        if Train_Method == "LBPH":
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='250%'>Method: LBPH Face Recognizer</span>"
        elif Train_Method == "Eigen":
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Eigen Face Recognizer</span>"
        elif Train_Method == "Fisher":
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Fisher Face Recognizer</span>"
        elif Train_Method == "ANN":
            title_text = "<span foreground='blue' font-size='25pt'>Method: Artificial Neural networks</span>"
        else:
            print "Error in Train Method"
            exit()
        self.face_recognizer.train(self.faces, np.array(self.labels_num))

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

        img = wx.Image(VideoWidth, VideoHeight)
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
            img = wx.Image(VideoWidth, VideoHeight)
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
                print "Error no webcam image"
            self.goButton.Enable()
            color = (255, 0, 0)
            self.BtnStartLiveVideo.SetBackgroundColour(color)
            self.BtnStartLiveVideo.SetLabel("Stop Live Video Stream")
        self.updateSizer()

    def startVideo(self, videopath=None):
        self.VideoFile = False
        if videopath == None: return
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
            aspectratio = float(VideoWidth) / float(w)
            self.frame = cv2.resize(self.frame, (VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
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
            img = wx.Image(VideoWidth, VideoHeight)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))
        self.updateSizer()

    def redraw(self, event):
        ret, self.frame = self.capture.read()
        if ret:
            if self.VideoFile:
                h, w = self.frame.shape[:2]
                aspectratio = float(VideoWidth) / float(w)
                self.frame = cv2.resize(self.frame, (VideoWidth, int(h * aspectratio)), cv2.INTER_LANCZOS4)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if self.detect_faces:
                detect_and_recognize_faces(image=self.frame, face_recognizer=self.face_recognizer, names=self.names,
                                           ann=self.ann)
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
            img = wx.Image(VideoWidth, VideoHeight)
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Face Recognizer")
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.faces, self.labels_num, self.names, self.ann = prepare_faces_data()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        title_text = " "
        imgText = "<span foreground='red' font-size='15pt'>Drag and Drop an Image.</span>"
        if Train_Method == "LBPH":
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='250%'>Method: LBPH Face Recognizer</span>"
        elif Train_Method == "Eigen":
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Eigen Face Recognizer</span>"
        elif Train_Method == "Fisher":
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
            title_text = "<span foreground='blue' font-size='25pt'>Method: Fisher Face Recognizer</span>"
        elif Train_Method == "ANN":
            title_text = "<span foreground='blue' font-size='25pt'>Method: Artificial Neural networks</span>"
        else:
            print "Error in Train Method"
            exit()
        self.face_recognizer.train(self.faces, np.array(self.labels_num))

        self.lblTitle = wx.StaticText(self.panel)
        self.lblTitle.SetLabelMarkup(title_text)
        self.firstLine = wx.StaticLine(self.panel, )
        self.lblDragandDrop = wx.StaticText(self.panel)
        self.lblDragandDrop.SetLabelMarkup(imgText)

        pub.subscribe(self.update_image_on_dnd, 'dnd')
        filedroptarget = DropTarget(self, False)

        bmp = wx.Image(VideoWidth, VideoHeight)
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
        self.img = scale_image(self.img, VideoWidth, VideoHeight)
        detect_and_recognize_faces(image=self.img, face_recognizer=self.face_recognizer, names=self.names, ann=self.ann)
        self.on_view()
        self.OKButton.Enable()

    def on_view(self, filepath=None):
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "New Face Input")
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.DataPath = Data_Path
        self.NumberImage = 0

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
        bmp = wx.Image(VideoWidth, VideoHeight)
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
        dirs = os.listdir(Data_Path)
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
        self.directory = Data_Path + "/" + self.Name
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence, r
        self.FaceImageName = self.directory + "/" + str(self.NumberImage) + ".png"
        self.InfoFaceName = self.directory + "/" + str(self.NumberImage) + ".info"
        file = open(self.InfoFaceName, "w+")
        for item in r:
            file.write("%s\n" % item)
        file.close()
        cv2.imwrite(self.FaceImageName, self.roi_gray)
        self.NumberImage = self.NumberImage + 1
        self.saveButton.Disable()

    def update_image_on_dnd(self, filepath):
        self.img = cv2.imread(filepath)
        if self.img is None: return
        self.img = scale_image(self.img, VideoWidth, VideoHeight)
        self.img, self.roi_gray, self.detect, self.points = detect_face(image=self.img)
        if dlib_detector:
            #
            # Draw the face landmarks on the screen.
            #
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
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Settings Input")
        loadSettings()
        self.LoadSettings = False
        self.width = ['1024', '800', '640', '480']
        self.lblList = ["16:9", "4:3"]
        self.methods = ['LBPH', 'Eigen', 'Fisher', 'ANN']
        self.verbose = ['Silent', 'Output All']
        self.Text1 = "Video Size: " + str(VideoWidth) + "x" + str(VideoHeight)
        self.Tverbose = "Output All"
        if yverbose:
            self.Tverbose = "Output All"
        else:
            self.Tverbose = "Silent"
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.lblSettingsFile = wx.StaticText(self.panel, label="Settings file: ")
        self.SettingsFile = wx.TextCtrl(self.panel, value="Default_Settings.config")
        self.BtnSettingsFile = GenButton(self.panel, label="Load")
        self.FileName = str(self.SettingsFile.GetValue())
        self.lblPhotosPath = wx.StaticText(self.panel, label="Data Path:     ")
        self.DataPath = wx.TextCtrl(self.panel, value=Data_Path)
        self.DataPathText = str(self.DataPath.GetValue())
        self.DataPath.Disable()
        self.BtnDataPath = GenButton(self.panel, label="Load")
        self.firstLine = wx.StaticLine(self.panel, )
        self.cmbText = wx.StaticText(self.panel, -1, "Select Aspect Ratio: ")
        self.AspectRatio = wx.RadioBox(self.panel, label='', choices=self.lblList, majorDimension=1,
                                       style=wx.RA_SPECIFY_ROWS)
        self.AspectRatio.SetStringSelection(AspectRatio)
        self.Aspect = str(self.AspectRatio.GetStringSelection())
        self.selectWidthText = wx.StaticText(self.panel, -1, "Select Width: ")
        self.cmbWidth = wx.ComboBox(self.panel, -1, "800", choices=self.width)
        self.cmbWidth.SetStringSelection(str(VideoWidth))
        self.sizeText1 = wx.StaticText(self.panel, -1, self.Text1)
        self.secondLine = wx.StaticLine(self.panel, )
        self.selectMethodText = wx.StaticText(self.panel, -1, "Select Method: ")
        self.cmbMethod = wx.ComboBox(self.panel, -1, "LBPH", choices=self.methods)
        self.cmbMethod.SetStringSelection(Train_Method)
        self.lblMaxConfidence = wx.StaticText(self.panel, label="Max Confidence: ")
        self.MaxConfidence = wx.TextCtrl(self.panel, value="75.0")
        self.MaxConfidence.SetLabel(str(MaxConfidence))
        self.selectVerboseText = wx.StaticText(self.panel, -1, "Output option: ")
        self.cmbVerbose = wx.ComboBox(self.panel, -1, "Silent", choices=self.verbose)
        self.cmbVerbose.SetStringSelection(self.Tverbose)
        self.thirdLine = wx.StaticLine(self.panel, )
        self.saveButton = GenButton(self.panel, label="Save")
        self.saveButton.Disable()
        self.closeButton = GenButton(self.panel, label="Cancel")

        self.BtnSettingsFile.Bind(wx.EVT_BUTTON, self.onSettingsFile)
        self.SettingsFile.Bind(wx.EVT_TEXT, self.onSettingsFileChange)
        self.BtnDataPath.Bind(wx.EVT_BUTTON, self.onDataPath)
        self.AspectRatio.Bind(wx.EVT_RADIOBOX, self.onAspectRatio)
        self.cmbWidth.Bind(wx.EVT_COMBOBOX, self.oncmbWidth)
        self.cmbMethod.Bind(wx.EVT_COMBOBOX, self.oncmbMethod)
        self.cmbVerbose.Bind(wx.EVT_COMBOBOX, self.oncmbVerbose)
        self.MaxConfidence.Bind(wx.EVT_TEXT, self.onMaxConfidence)
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)
        self.closeButton.Bind(wx.EVT_BUTTON, self.onCancel)

        self.getVideoDimensions()

        self.panel.SetSizer(self.topSizer)
        self.topSizer.Fit(self)
        self.panel.Center(wx.BOTH)
        self.Center(wx.BOTH)
        self.Show()

    def UpdateSizer(self):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
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
        selectwidthSizer.Add(self.AspectRatio, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.selectWidthText, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.cmbWidth, 0, wx.ALL, 5)
        selectwidthSizer.Add(self.sizeText1, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.selectMethodText, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.cmbMethod, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.lblMaxConfidence, 0, wx.ALL, 5)
        selectmethodSizer.Add(self.MaxConfidence, 1, wx.ALL | wx.EXPAND, 5)
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
        self.Refresh()

    def getVideoDimensions(self):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        self.Aspect = str(self.AspectRatio.GetStringSelection())
        heightA = ['576', '450', '360', '270']
        heightB = ['768', '600', '480', '360']
        if self.AspectRatio.GetStringSelection() == "16:9":
            self.VideoWidth = self.width[self.width.index(self.cmbWidth.GetValue())]
            self.VideoHeight = heightA[self.width.index(self.cmbWidth.GetValue())]
        elif self.AspectRatio.GetStringSelection() == "4:3":
            self.VideoWidth = self.width[self.width.index(self.cmbWidth.GetValue())]
            self.VideoHeight = heightB[self.width.index(self.cmbWidth.GetValue())]
        self.Text1 = "Video Size: " + str(self.VideoWidth) + "x" + str(self.VideoHeight)
        self.sizeText1.SetLabel(self.Text1)
        self.sizeText1.Update()
        self.sizeText1.Refresh()
        AspectRatio = self.AspectRatio.GetStringSelection()
        VideoWidth = int(self.VideoWidth)
        VideoHeight = int(self.VideoHeight)
        self.UpdateSizer()
        return

    def onCancel(self, event):
        self.Close()

    def onSave(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        writeSettings(self.FileName)
        self.saveButton.Disable()

    def onMaxConfidence(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        if self.LoadSettings: return
        MaxConfidence = float(self.MaxConfidence.GetValue())
        self.saveButton.Enable()

    def oncmbMethod(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        if self.LoadSettings: return
        Train_Method = self.cmbMethod.GetStringSelection()
        self.saveButton.Enable()

    def oncmbVerbose(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        if self.LoadSettings: return
        self.Tverbose = self.cmbVerbose.GetStringSelection()
        if self.Tverbose == "Output All":
            yverbose = True
        elif self.Tverbose == "Silent":
            yverbose = False
        self.saveButton.Enable()

    def oncmbWidth(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        if self.LoadSettings: return
        self.getVideoDimensions()
        self.saveButton.Enable()

    def onAspectRatio(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        if self.LoadSettings: return
        self.getVideoDimensions()
        self.saveButton.Enable()

    def onSettingsFile(self, event):
        openFileDialog = wx.FileDialog(self, "Open", "", "", "Settings files (*.config)|*.config",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.SetDirectory("./")
        openFileDialog.ShowModal()
        self.SettingsFile.SetValue(openFileDialog.GetPath())
        openFileDialog.Destroy()

    def onSettingsFileChange(self, event):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        self.LoadSettings = True
        self.FileName = str(self.SettingsFile.GetValue())
        loadSettings()
        self.DataPathText = Data_Path
        self.Aspect = AspectRatio
        self.VideoWidth = VideoWidth
        self.VideoHeight = VideoHeight
        self.DataPath.SetValue(self.DataPathText)
        self.AspectRatio.SetStringSelection(self.Aspect)
        self.getVideoDimensions()
        self.Refresh()
        self.LoadSettings = False

    def onDataPath(self, event):
        openDirDialog = wx.DirDialog(self, "Choose Data directory", "",
                                     wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        openDirDialog.SetPath("./")
        openDirDialog.ShowModal()
        self.DataPath.SetValue(openDirDialog.GetPath())
        self.DataPathText = str(self.DataPath.GetValue())
        self.saveButton.Enable()
        openDirDialog.Destroy()


###########################################################################
## Class MainWindow
###########################################################################

class MainWindow(wx.Frame):

    def __init__(self, parent, *args, **kwargs):
        global Data_Path, AspectRatio, VideoWidth, VideoHeight, yverbose, Train_Method, MaxConfidence
        super(MainWindow, self).__init__(*args, **kwargs)
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                          size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.panel = wx.Panel(self)
        loadSettings()
        self.InitUI()

    def InitUI(self):
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        SizerVertical = wx.BoxSizer(wx.VERTICAL)

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
        self.Bind(wx.EVT_MENU, self.onQuit, exitMenuItem)
        self.Bind(wx.EVT_MENU, self.onSettings, configureMenuItem)
        self.Bind(wx.EVT_MENU, self.onNewFace, addFaces)
        self.Bind(wx.EVT_MENU, self.onNewFaceFromVideo, addFacesFromVideo)
        self.Bind(wx.EVT_MENU, self.onRecognizeFace, recognizeFaces)
        self.Bind(wx.EVT_MENU, self.onRecognizeFaceInVideo, recognizeFacesinVideo)
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(AddFacesMenu, "&Add Faces")
        menuBar.Append(RecognizeFacesMenu, "&Recognize Faces")
        self.SetMenuBar(menuBar)

        self.ToolBar = wx.ToolBar(self, wx.ID_ANY)
        self.ToolBar.SetToolBitmapSize((50, 50))
        self.Exit = self.ToolBar.AddTool(wx.ID_ANY, u"Exit", wx.Bitmap(u"icons/icons8-exit.png"))
        self.Configure = self.ToolBar.AddTool(wx.ID_ANY, u"Configure", wx.Bitmap(u"icons/icons8-settings.png"))
        self.ToolBar.AddSeparator()
        self.New_Face = self.ToolBar.AddTool(wx.ID_ANY, u"New Face", wx.Bitmap(u"icons/icons8-add_face.png"))
        self.New_Face_from_Video = self.ToolBar.AddTool(wx.ID_ANY, u"New Face from Video",
                                                        wx.Bitmap(u"icons/icons8-add_face_from_video.png"))
        self.ToolBar.AddSeparator()
        self.Recognize_Face = self.ToolBar.AddTool(wx.ID_ANY, u"Recognize Face",
                                                   wx.Bitmap(u"icons/icons8-recognize_face.png"))
        self.Recognize_Face_in_Video = self.ToolBar.AddTool(wx.ID_ANY, u"Recognize Face from Video",
                                                            wx.Bitmap(u"icons/icons8-recognize_in_video.png"))
        self.ToolBar.Realize()

        self.Bind(wx.EVT_TOOL, self.onQuit, self.Exit)
        self.Bind(wx.EVT_TOOL, self.onSettings, self.Configure)
        self.Bind(wx.EVT_TOOL, self.onNewFace, self.New_Face)
        self.Bind(wx.EVT_TOOL, self.onNewFaceFromVideo, self.New_Face_from_Video)
        self.Bind(wx.EVT_TOOL, self.onRecognizeFace, self.Recognize_Face)
        self.Bind(wx.EVT_TOOL, self.onRecognizeFaceInVideo, self.Recognize_Face_in_Video)

        SizerVertical.Add(self.ToolBar, 0, wx.EXPAND, 5)

        self.SetSizer(SizerVertical)
        self.Layout()

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
