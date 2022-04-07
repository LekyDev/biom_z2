import os
import cv2
import numpy as np
import dlib
from shapely.geometry import Polygon
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import mean_squared_error


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

# Detecting face (Viola-Jones)
def ViolaJones(img, bounding_box):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2)

    truePositives = 0
    falsePositives = 0
    falseNegatives = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceROI = img[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        iou = calculate_iou([x, y, x + w, y + h], bounding_box)

        if iou > 0.5 and truePositives == 0:
            truePositives += 1
        elif iou > 0.5 and truePositives > 0:
            falsePositives += 1
        elif iou < 0.5:
            falseNegatives += 1

        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            img = cv2.circle(img, eye_center, radius, (255, 0, 0), 4)

        if (truePositives + falsePositives) != 0:
            recall = truePositives / (truePositives + falseNegatives)
        else:
            recall = 0

        if(truePositives + falsePositives) != 0:
            precision = truePositives / (truePositives + falsePositives)
        else:
            precision = 0

        img = cv2.putText(img, "Precision: " + str(precision), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1, cv2.LINE_AA, False)
        img = cv2.putText(img, "Recall:" + str(recall), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1, cv2.LINE_AA, False)

# Calculating intersection over union (IOU) of two rectangles
def calculate_iou(rectangle1, rectangle2):
    rectangle1 = [int(i) for i in rectangle1]
    rectangle2 = [int(i) for i in rectangle2]
    box_1 = [[rectangle1[0], rectangle1[1]], [rectangle1[0] + rectangle1[2], rectangle1[1]],
             [rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]],
             [rectangle1[0], rectangle1[1] + rectangle1[3]]]
    box_2 = [[rectangle2[0], rectangle2[1]], [rectangle2[0] + rectangle2[2], rectangle2[1]],
             [rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]],
             [rectangle2[0], rectangle2[1] + rectangle2[3]]]
    # X Y W H
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def CNN(img, bounding_box, left_eye_ground, right_eye_ground, quality, frame, videoName):
    left_eye_ground = np.array(left_eye_ground)
    right_eye_ground = np.array(right_eye_ground)
    eye_array = np.column_stack((centeroidnp(left_eye_ground), centeroidnp(right_eye_ground)))

    detector = MTCNN()

    location = detector.detect_faces(img)
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    if len(location) > 0:
        for face in location:
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 4)
            cv2.circle(img, face['keypoints']['left_eye'], 1, (0, 0, 255), 4)
            cv2.circle(img, face['keypoints']['right_eye'], 1, (0, 0, 255), 4)
            iou = calculate_iou([x, y, x2, y2], bounding_box)
            if(face['keypoints']['left_eye'] is not None and face['keypoints']['right_eye'] is not None):
                mse_calc = mean_squared_error(eye_array, [face['keypoints']['left_eye'], face['keypoints']['right_eye']])
            else:
                mse_calc = 0


        if mse_calc>200:
                print('-------------')
                print('Video name: ' + videoName)
                print('Quality: ' + quality)
                print('Frame: ' + str(frame))
                print('MSE: ' + str(mse_calc))
                print('-------------')

        if iou > 0.5 and truePositives == 0:
            truePositives += 1
        elif iou > 0.5 and truePositives > 0:
            falsePositives += 1
        elif iou < 0.5:
            falseNegatives += 1

        if (truePositives + falsePositives) != 0:
            recall = truePositives / (truePositives + falseNegatives)
        else:
            recall = 0

        if(truePositives + falsePositives) != 0:
            precision = truePositives / (truePositives + falsePositives)
        else:
            precision = 0

        img = cv2.putText(img, "Precision: " + str(precision), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1, cv2.LINE_AA, False)
        img = cv2.putText(img, "Recall:" + str(recall), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1, cv2.LINE_AA, False)
        img = cv2.putText(img, "MSE:" + str(round(mse_calc,2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 0, 255), 1, cv2.LINE_AA, False)

def files_to_videos(detector):
    folder = "video_input/"
    filenames = ["Kieran_Culkin_0.npz", "Liu_Ye_2.npz",
                 "Maggie_Smith_3.npz", "Margaret_Thatcher_5.npz",
                 "Marisa_Tomei_1.npz", "Martin_Sheen_3.npz",
                 "Oscar_Elias_Biscet_0.npz", "Natalie_Stewart_2.npz",
                 "Matt_Anderson_2.npz"]

    videoNumber = 0
    for filename in filenames:
        filepath = (os.path.join(folder, filename))
        videoFile = np.load(filepath)
        colorImages = videoFile['colorImages_original']
        colorImagesMedium = videoFile['colorImages_medium']
        colorImagesSevere = videoFile['colorImages_severe']
        boundingBox = videoFile['boundingBox']
        landmarks2D = videoFile['landmarks2D']
        videoNumber += 1
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        h = colorImages.shape[0]
        w = colorImages.shape[1]
        if (detector == 'Viola'):
            out = cv2.VideoWriter("C:/Users/User/Downloads/biom2/viola/video" + str(videoNumber) + ".avi", fourcc, 10, (w * 4, h), isColor=True)
        else:
            out = cv2.VideoWriter("C:/Users/User/Downloads/biom2/cnn/video" + str(videoNumber) + ".avi", fourcc, 10, (w * 4, h), isColor=True)
        for i in range(colorImages.shape[-1]):
            left_eye = []
            left_eye.append(landmarks2D[37, :, i])
            left_eye.append(landmarks2D[38, :, i])
            left_eye.append(landmarks2D[39, :, i])
            left_eye.append(landmarks2D[40, :, i])
            left_eye.append(landmarks2D[41, :, i])
            left_eye.append(landmarks2D[41, :, i])
            right_eye = []
            right_eye.append(landmarks2D[43, :, i])
            right_eye.append(landmarks2D[44, :, i])
            right_eye.append(landmarks2D[45, :, i])
            right_eye.append(landmarks2D[46, :, i])
            right_eye.append(landmarks2D[47, :, i])
            right_eye.append(landmarks2D[48, :, i])
            img0 = np.zeros((h, w, 3), np.uint8)
            img0 = cv2.rectangle(img0, (int(boundingBox[0][0][i]), int(boundingBox[0][1][i])), (int(boundingBox[3][0][i]), int(boundingBox[3][1][i])), (255, 0, 0), 2)
            for j in range(landmarks2D.shape[0]):
                img0 = cv2.circle(img0, (int(landmarks2D[j][0][i]), int(landmarks2D[j][1][i])), 2, (0, 255, 0), thickness=-1)
            img1 = colorImages[:, :, :, i]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            if(detector=='Viola'):
                #print([landmarks2D[:, :, i], landmarks2D[:, 1, i]])
                img1 = cv2.circle(img1, (int(landmarks2D[38][0][i]), int(landmarks2D[38][1][i])), 2, (0, 255, 0),
                                  thickness=-1)
                ViolaJones(img1, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]), int(boundingBox[3][1][i])])
            else:
                CNN(img1, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]), int(boundingBox[3][1][i])], left_eye, right_eye, "original", i, "C:/Users/User/Downloads/biom2/cnn/video" + str(videoNumber) + ".avi")
            img2 = colorImagesMedium[:, :, :, i]
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if (detector == 'Viola'):
                ViolaJones(img2, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]), int(boundingBox[3][1][i])])
            else:
                CNN(img2, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]), int(boundingBox[3][1][i])], left_eye, right_eye, "medium", i, "C:/Users/User/Downloads/biom2/cnn/video" + str(videoNumber) + ".avi")
            img3 = colorImagesSevere[:, :, :, i]
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            if (detector == 'Viola'):
                ViolaJones(img3, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]), int(boundingBox[3][1][i])])
            else:
                CNN(img3, [int(boundingBox[0][0][i]), int(boundingBox[0][1][i]), int(boundingBox[3][0][i]),int(boundingBox[3][1][i])], left_eye, right_eye, "severe", i, "C:/Users/User/Downloads/biom2/cnn/video" + str(videoNumber) + ".avi")
            oneForAll = np.hstack((img0, img1, img2, img3))
            out.write(oneForAll)
        out.release()


files_to_videos('Viola')
