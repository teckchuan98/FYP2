import cv2
import numpy as np
from Code_Deliverables.Compute_detections import predict
import time

# function takes frame, ort session, input name and returns frame with detected faces
def detect(frame, ort_session, input_name):
    h, w, _ = frame.shape

    # preprocess img acquired
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.99)

    for i in range(boxes.shape[0]):
        box = boxes[i]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)


   # cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    frame = cv2.resize(frame, (w, h))  # resize

    cv2.imshow('Video', frame)
    return frame, boxes.shape[0]