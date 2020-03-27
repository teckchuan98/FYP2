import numpy as np
import pickle
import cv2
import onnxruntime as ort
from detector import detect
from collections import Counter


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    video_capture = cv2.VideoCapture('chandler.mp4')
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)
    while True:
        ret, frame = video_capture.read()

        if frame is not None:

            boxes, labels, probs = detect(frame, ort_session, input_name)
            face_locations = []
            for i in boxes:
                x1, y1, x2, y2 = i
                y = (x1, y1, x2, y2)
                face_locations.append(y)

            face_encodings = []
            for i in face_locations:
                (X1, Y1, X2, Y2) = i
                face = frame[Y1:Y2, X1:X2]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                face_encodings.append(vec.flatten())

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                diff = np.subtract(saved_embeds, face_encoding)
                dist = np.sum(np.square(diff), axis=1)

                rr = []

                for distance in dist:
                    if distance < 0.55:
                        rr.append(True)
                    else:
                        rr.append(False)

                new_name = []
                for result in range(len(rr)):
                    if rr[result]:
                        new_name.append(names[result])

                print(new_name)
                if len(new_name) > 0:
                    person = Counter(new_name).most_common(1)
                    face_names.append(person[0][0])
                else:
                    face_names.append("unknown")




                #idx = np.argmin(dist)

                #if dist[idx] < 0.4:
                    #face_names.append(names[idx])
                #else:
                    #face_names.append("unknown")

            for (x1, y1, x2, y2), name in zip(face_locations, face_names):
                if name == "unknown":
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
