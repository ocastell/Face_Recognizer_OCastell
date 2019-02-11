import math

import cv2
import dlib
import numpy as np

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
video = False
predictor_model = "./dlib/shape_predictor_68_face_landmarks.dat"
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
global pupils_mesures, pupil_distance
pupils_mesures = 0
pupil_distance = []


def dlib_to_cv2_rectangle(face_rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.right() - x
    h = face_rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def getdistance(point_A=None, point_B=None):
    # if point_A and point_B == None: return
    a = np.array(point_A)
    b = np.array(point_B)
    distance = np.linalg.norm(a - b)
    return distance


def getmiddlepoint(point_A=None, point_B=None):
    # if point_A and point_B == None: return
    a = np.array(point_A)
    b = np.array(point_B)
    middlepoint = (a + b) / 2
    return (middlepoint[0], middlepoint[1])


def get_face_metric(points, pupil_distance=0):
    if pupil_distance == 0: return
    interocular_distance = int(abs(getdistance(points[39], points[42])))
    interpupil_distance = int(pupil_distance)
    left_eye_width = int(abs(getdistance(points[45], points[42])))
    left_eye_height = int(
        abs(getdistance(getmiddlepoint(points[43], points[44]), getmiddlepoint(points[47], points[46]))))
    right_eye_width = int(abs(getdistance(points[36], points[39])))
    right_eye_height = int(
        abs(getdistance(getmiddlepoint(points[37], points[38]), getmiddlepoint(points[41], points[40]))))
    conversion = 64. / float(interpupil_distance)
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
    r1 = float(nose_width) / float(nose_height)
    r2 = float(left_eye_width) / float(left_eye_height)
    r3 = float(right_eye_width) / float(right_eye_height)
    r4 = float(getdistance(points[27], points[0])) / float(nose_height)
    r5 = float(getdistance(points[27], points[16])) / float(nose_height)
    r6 = float(getdistance(points[27], points[0])) / float(getdistance(points[31], points[36]))
    r7 = float(getdistance(points[27], points[16])) / float(getdistance(points[35], points[45]))
    r8 = float(getdistance(points[31], points[36])) / float(getdistance(points[31], points[27]))
    r9 = float(getdistance(points[45], points[35])) / float(getdistance(points[35], points[27]))
    r10 = fwhr
    #
    # Print Info
    #
    print " Interocular distance (pixels, mm): ", interocular_distance, int(interocular_distance * conversion)
    print " Interpupil distance (pixels, mm): ", interpupil_distance, int(interpupil_distance * conversion)
    print " Left eye width (pixels, mm): ", left_eye_width, int(left_eye_width * conversion)
    print " Left eye height (pixels, mm): ", left_eye_height, int(left_eye_height * conversion)
    print " Right eye width (pixels, mm): ", right_eye_width, int(right_eye_width * conversion)
    print " Right eye height (pixels, mm): ", right_eye_height, int(right_eye_height * conversion)
    print " Nose height (pixels, mm): ", nose_height, int(nose_height * conversion)
    print " Nose width (pixels, mm): ", nose_width, int(nose_width * conversion)
    print " Upper lip height (pixels, mm): ", upper_lip_height, int(upper_lip_height * conversion)
    print " Lower lip height (pixels, mm): ", lower_lip_height, int(lower_lip_height * conversion)
    print " Lip height (pixels, mm): ", lip_height, int(lip_height * conversion)
    print " Facial width (pixels, mm): ", facial_width, int(facial_width * conversion)
    print " Upper Facial height (pixels, mm): ", upper_facial_height, int(upper_facial_height * conversion)
    print " Lower Facial height (pixels, mm): ", lower_facial_height, int(lower_facial_height * conversion)
    print " Facial height (pixels, mm): ", int(facial_height), int(facial_height * conversion)
    print " Jaw width (pixels, mm): ", jaw_width, int(jaw_width * conversion)
    print " Facial-Width-Height ratio is: ", fwhr
    return


def pupils_tracking2(points, img):
    #
    # ROI eyes
    #
    image = img
    max_left_eye_y = max(points[47][1], points[46][1])
    min_left_eye_y = min(points[43][1], points[44][1])
    max_right_eye_y = max(points[40][1], points[41][1])
    min_right_eye_y = min(points[37][1], points[38][1])
    x1_right_eye, y1_right_eye = (points[36][0], min_right_eye_y - 5)
    x2_right_eye, y2_right_eye = (points[39][0], max_right_eye_y + 5)
    x1_left_eye, y1_left_eye = (points[42][0], min_left_eye_y - 5)
    x2_left_eye, y2_left_eye = (points[45][0], max_left_eye_y + 5)
    left_eye_roi = image[y1_left_eye:y2_left_eye, x1_left_eye:x2_left_eye]
    right_eye_roi = image[y1_right_eye:y2_right_eye, x1_right_eye:x2_right_eye]
    (x, y) = getpupil_center2(left_eye_roi)
    #print x, y
    cv2.circle(left_eye_roi,(x,y),2,(127,255,200),-1)
    cv2.imshow("left",left_eye_roi)
    cv2.waitKey(0)
    x_left = x1_left_eye + x
    y_left = y1_left_eye + y
    (x, y) = getpupil_center2(right_eye_roi)
    print x, y
    cv2.circle(right_eye_roi,(x,y),2,(127,255,200),-1)
    cv2.imshow("right",right_eye_roi)
    cv2.waitKey(0)
    x_right = x1_right_eye + x
    y_right = y1_right_eye + y
    return (x_left,y_left,x_right,y_right)

def getpupil_center2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # smooth images
    #
    #value = (2, 2)
    #img = cv2.GaussianBlur(img, value, 0)
    img = cv2.bilateralFilter(img,9,60,60)
    img = cv2.bitwise_not(img)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
    print "minVal", minVal
    print "maxVal", maxVal
    print "minLoc", minLoc
    print "maxLoc", maxLoc
    cv2.circle(img,maxLoc,2,(127,255,200),-1)
    cv2.imshow("right",img)
    cv2.waitKey(0)
    #
    # images in float type
    #
    img_f = np.asarray(img)
    w, h = img_f.shape[:2]
    dots = []
    points = []
    gx, gy = np.gradient(img_f)
    init_h = maxLoc[0]-5
    end_h = maxLoc[0]+5
    init_w = maxLoc[1]-5
    end_w = maxLoc[1]+5
    if init_h < 0: init_h = 0
    if end_h > h: end_h = h
    if init_w < 0: init_w = 0
    if end_w > w: end_w = w
    for i in range(init_h, end_h):
        for j in range(init_w, end_w):
            center = (i, j)
            des_x, des_y = vector_displacement(img_f, center)
            dot_prod = dot_products(des_x, des_y, gx, gy)
            dots.append(dot_prod)
            points.append((i, j))
    maxi = max(dots)
    index = dots.index(maxi)
    punt = points[index]
    print "maxim al punt ", punt
    print "valor ", maxi
    (i, j) = punt
    print "valor image ", img[j][i]
    return (i, j)

def dot_products(displacement_x, displacement_y, gradient_x, gradient_y):
    w, h = gradient_x.shape[:2]
    ntotal = float(w * h)
    total = 0.0
    for i in range(0, w):
        for j in range(0, h):
            sum = (displacement_x[i][j] * gradient_x[i][j])
            sum = sum + (displacement_y[i][j] * gradient_y[i][j])
            total = total + (sum ** 2)
    total = total / ntotal
    return total


def vector_displacement(img, center=(0, 0)):
    w, h = img.shape[:2]
    displacement_x = np.zeros((w, h))
    displacement_y = np.zeros((w, h))
    (center_x, center_y) = center
    for i in range(0, w):
        for j in range(0, h):
            val1 = float(i) - float(center_x)
            val2 = float(j) - float(center_y)
            modul = math.sqrt(val1 ** 2 + val2 ** 2)
            if modul == 0:
                displacement_x[i][j] = 0.0
                displacement_y[i][j] = 0.0
            else:
                displacement_x[i][j] = val1 / modul
                displacement_y[i][j] = val2 / modul
    return displacement_x, displacement_y

def deskew(img,SZ,affine_flags):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def pupils_tracking(points, img):
    SZ = 20
    bin_n = 16
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    detect = False
    x_left_eye=0
    y_left_eye=0
    x_right_eye=0
    y_right_eye=0
    #
    # Image to analize
    #
    image = img
    #
    # position of the right anf left eyes from dlib landmarks
    #
    max_left_eye_y = max(points[47][1], points[46][1])
    min_left_eye_y = min(points[43][1], points[44][1])
    max_right_eye_y = max(points[40][1], points[41][1])
    min_right_eye_y = min(points[37][1], points[38][1])
    x1_right_eye, y1_right_eye = (points[36][0], min_right_eye_y - 5)
    x2_right_eye, y2_right_eye = (points[39][0], max_right_eye_y + 5)
    x1_left_eye, y1_left_eye = (points[42][0], min_left_eye_y - 5)
    x2_left_eye, y2_left_eye = (points[45][0], max_left_eye_y + 5)
    #
    # processing the ROI images
    #
    left_eye_roi = image[y1_left_eye:y2_left_eye, x1_left_eye:x2_left_eye]
    right_eye_roi = image[y1_right_eye:y2_right_eye, x1_right_eye:x2_right_eye]
    #
    # Process the left eye
    #
    left_eye_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
    img_deskew  = deskew(left_eye_gray,SZ,affine_flags)
    cv2.imshow("left eye",img_deskew)
    cv2.waitKey(0)
    detect, (x, y) = getpupil_center(left_eye_gray)
    if detect:
       x_left_eye = x1_left_eye + x
       y_left_eye = y1_left_eye + y
    #cv2.circle(left_eye_roi,(x,y),2,(127,255,200),-1)
    #cv2.imshow("left eye",left_eye_roi)
    #cv2.waitKey(0)
    #
    # Process the right eye
    #
    right_eye_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
    detect, (x, y) = getpupil_center(right_eye_gray)
    if detect:
       x_right_eye = x1_right_eye + x
       y_right_eye = y1_right_eye + y
    #cv2.circle(right_eye_roi,(x,y),2,(127,255,200),-1)
    #cv2.imshow("right eye",right_eye_roi)
    #cv2.waitKey(0)
    return detect, (x_left_eye, y_left_eye, x_right_eye, y_right_eye)

def getpupil_center(img):
    detect = False
    (x, y) = (0, 0)
    img = cv2.bilateralFilter(img,17,55,55)
    img_inv = cv2.bitwise_not(img)
    #cv2.imshow("img",img_inv)
    #cv2.waitKey(0)
    circles = cv2.HoughCircles(img_inv, cv2.HOUGH_GRADIENT, 2., 200,
                               param1=35, param2=15, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 1:
            print "problem in pupils_tracking"
            exit()
        for (x, y, r) in circles:
            #print x, y, r
            detect = True
            break
    return detect,(x,y)

def principal(detected_faces):
    global pupils_mesures, pupil_distance
    for i, face_rect in enumerate(detected_faces):
        if not video:
            print(
            "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                               face_rect.right(), face_rect.bottom()))
        # Draw a box around each face we found
        (x, y, w, h) = dlib_to_cv2_rectangle(face_rect)
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Get the the face's landmarks
        pose_landmarks = face_pose_predictor(image, face_rect)
        points = shape_to_np(pose_landmarks)
        # Get the center of the pupils
        detect = False
        detect, (x1, y1, x2, y2) = pupils_tracking(points, image)
        #(x1, y1, x2, y2) = pupils_tracking2(points, image)
        if video:
            if x1 <> 0 and y1 <> 0 and x2 <> 0 and y2 <> 0 and pupils_mesures <= 100:
                if detect:
                   pupils_mesures = pupils_mesures + 1
                   dist = getdistance((x1, y1), (x2, y2))
                   pupil_distance.append(dist)
            if pupils_mesures == 100:
                max_pupil_distance = max(pupil_distance)
                min_pupil_distance = min(pupil_distance)
                pupil_distance_average = sum(pupil_distance) / float(len(pupil_distance))
                get_face_metric(points, pupil_distance=pupil_distance_average)
                pupils_mesures = 0
        else:
            if x1 <> 0 and y1 <> 0 and x2 <> 0 and y2 <> 0 :
                if detect:
                   pupil_distance_average = getdistance((x1, y1), (x2, y2))
                   get_face_metric(points, pupil_distance=pupil_distance_average)
                else:
                   print "pupils don't detected "
            else:
                print "error in pupils "
                # exit()
        # Draw the face landmarks on the screen.
        color = (0, 0, 255)
        for (x, y) in points:
            cv2.circle(image, (x, y), 1, color, -1)
        if x1 <> 0 and y1 <> 0 and x2 <> 0 and y2 <> 0:
            cv2.circle(image, (x1, y1), 5, (0, 255, 0), 1)
            cv2.circle(image, (x2, y2), 5, (0, 255, 0), 1)
            cv2.rectangle(image, (x1 - 1, y1 - 1), (x1 + 1, y1 + 1), (0, 128, 255), -1)
            cv2.rectangle(image, (x2 - 1, y2 - 1), (x2 + 1, y2 + 1), (0, 128, 255), -1)
        cv2.imshow("landmarks", image)
    if not video:
        cv2.waitKey(0)


if not video:
    file_name = "./prova2.jpg"
    image = cv2.imread(file_name)
    detected_faces = face_detector(image, 1)
    print("Found {} faces in video ".format(len(detected_faces)))
    principal(detected_faces)
else:
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    while (capture.isOpened()):
        ret, image = capture.read()
        detected_faces = face_detector(image, 1)
        if ret:
            principal(detected_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
