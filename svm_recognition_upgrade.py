import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect
import time

def track(pre_faces, cur_faces):
    print("in function")
    print(pre_faces, cur_faces)
    results = []
    for face in pre_faces:
        min_dif = None
        min_id = -1
        for i in range(len(cur_faces)):
            face1 = cur_faces[i]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            if(min_dif == None or min_dif > dif):
                min_dif = dif
                min_id = i
        if(min_id != -1):
            results.append(cur_faces[min_id])
        else:
            break

    return results

def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    recognizer = pickle.loads(open("recognizer.pkl", "rb").read())
    le = pickle.loads(open("le.pkl", "rb").read())

    video_capture = cv2.VideoCapture('chandler.mp4')
    #with open("embeddings.pkl", "rb") as f:
        #(saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))
    fps = 0.0
    redetect = -1
    face_locations = []
    face_names = []
    probability = []

    while True:
        redetect = (redetect+1)%20
        ret, frame = video_capture.read()
        start = time.time()
        # frame = cv2.resize(frame, (320, 240))

        if frame is not None:

            boxes, labels, probs = detect(frame, ort_session, input_name)
            temp = []
            for i in boxes:
                x1, y1, x2, y2 = i
                y = (y1, x2, y2, x1)
                temp.append(y)
            rgb_frame = frame[:, :, ::-1]

            if redetect == 0:
                face_locations = temp
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_names = []
                probability = []

                for face_encoding in face_encodings:

                    face_encoding = [face_encoding]
                    preds = recognizer.predict_proba(face_encoding)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    if proba > 0.5:
                        face_names.append(name)
                        probability.append(proba)
                    else:
                        face_names.append("unknown")

            else:
                face_locations = track(face_locations, temp)

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

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            fps = (fps + (1. / (time.time() - start))) / 2
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
