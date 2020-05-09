import cv2
import numpy as np
from modular.Compute_detections import predict
import face_recognition
import onnxruntime as ort
import pickle

def update(frame, pre_frame, face_location):
    return None


def initialise():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    recognizer = pickle.loads(open("recognizer.pkl", "rb").read())
    le = pickle.loads(open("le.pkl", "rb").read())
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    video_capture = cv2.VideoCapture('chandler.mp4')

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))

    return ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out



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

    temp = []
    for i in boxes:
        x1, y1, x2, y2 = i
        y = (y1, x2, y2, x1)
        temp.append(y)
    rgb_frame = frame[:, :, ::-1]

    return rgb_frame, temp


def recognise(temp, rgb_frame, recognizer, le, names, saved_embeds):
    face_locations = temp
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    probability = []
    for face_encoding in face_encodings:
        diff = np.subtract(saved_embeds, face_encoding)
        dist = np.sum(np.square(diff), axis=1)
        idx = np.argmin(dist)

        if dist[idx] < 0.29:
            id = names[idx]
        else:
            id = "unknown"

        face_encoding = [face_encoding]
        preds = recognizer.predict_proba(face_encoding)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        if proba > 0.6 and name == id:
            face_names.append(name)
            probability.append(proba)
        elif proba > 0.7 and id == "unknown" and name != "unknown":
            face_names.append(name)
            probability.append(proba)
        elif id != "unknown" and name == "unknown":
            face_names.append(id)
            probability.append(proba)
        else:
            face_names.append("unknown")
            probability.append(proba)

    return face_locations, face_names, probability


def track(pre_faces, cur_faces, names, frame, pre_frame):
    print("in function")
    print(pre_faces, cur_faces)
    results = []
    results_names = []

    for n in range(len(pre_faces)):
        face = pre_faces[n]
        min_dif = None
        min_id = -1
        for i in range(len(cur_faces)):
            face1 = cur_faces[i]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            if dif <= 30:
                if (min_dif == None or min_dif > dif):
                    min_dif = dif
                    min_id = i
        if(min_id != -1):
            results.append(cur_faces[min_id])
            results_names.append(names[n])
        else:
            updated_loc = update(frame, pre_frame, pre_faces[n])
            if updated_loc is not None:
                results.append(cur_faces[min_id])
                results_names.append(names[n])

    return results, results_names


def tag(frame, face_locations, face_names, probability):
    for (top, right, bottom, left), name, prob in zip(face_locations, face_names, probability):
        if name == "unknown":
            continue
        x = prob * 100
        x = str(x)
        x = x[:3]
        x = x + "%"
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name + " : " + x, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return frame









