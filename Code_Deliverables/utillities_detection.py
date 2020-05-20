import cv2
import numpy as np
from Code_Deliverables.Compute_detections import predict
import onnxruntime as ort



def initialiseDetector():
    """
    Description : This function loads detection model
    Author : Jeetun Ishan
    Last modified : 20/05/2020
    param : none
    Return :
            ort_seesion: The initialised face detector model
    """
    ort_session = ort.InferenceSession('models/ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    return ort_session, input_name


def detect(frame, ort_session, input_name):
    """
    Description : This function takes the frame, ort session, input name and returns an rgb frame and a list of detected
                  face locations
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    params :
            frame: The video frame to perform detection
            ort-session: The onnx detection model
            input_name: input name of frame
    Return :
            rgb_frame: the frame converted to rgb format
            temp: List of detected faces in x1, y1, x2, y2 format
    Reference:
            https://github.com/fyr91/face_detection/blob/master/detect_ultra_light.py
    """

    #
    h, w, _ = frame.shape

    # processing image starts here
    # referenced from https://github.com/fyr91/face_detection/blob/master/detect_ultra_light.py
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # run the detection model on current frame
    confidences, boxes = ort_session.run(None, {input_name: img})

    # the predict function will predict all boxes with human faces by using an IOU threshold
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

    temp = []
    # changing the coordinates position as required for face recognition function
    for i in boxes:
        x1, y1, x2, y2 = i
        y = (y1, x2, y2, x1)
        temp.append(y)
    rgb_frame = frame[:, :, ::-1]

    return rgb_frame, temp