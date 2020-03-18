import cv2
import numpy as np
from Compute_detections import predict
import dlib


def detect(frame, ort_session, input_name):
    h, w, _ = frame.shape
    # pre-process img acquired
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    return boxes, labels, probs


def align(box, frame, fa):
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aligned_face = fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
    aligned_face = cv2.resize(aligned_face, (112, 112))
    aligned_face = aligned_face - 127.5
    aligned_face = aligned_face * 0.0078125
    return aligned_face


def compute_embeddings(faces, images_placeholder, sess, phase_train_placeholder, embeddings):
    faces = np.array(faces)
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    embeds = sess.run(embeddings, feed_dict=feed_dict)
    return embeds


def draw(boxes, predictions, frame):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        text = str(predictions[i])
        if text != "unknown":
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return frame
