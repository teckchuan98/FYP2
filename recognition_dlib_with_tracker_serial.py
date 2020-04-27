import sys

import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect
from random import randint
import numpy as np
import time

def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    video_capture = cv2.VideoCapture("zoom.mp4")
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))

    ok, frame = video_capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    trackers = []
    colors = []
    bboxes, labels, probs = detect(frame, ort_session, input_name)
    print(bboxes)
    for bbox in bboxes:
        colors.append((randint(0, 255)))
        tracker = cv2.TrackerKCF_create()
        bbox = tuple(bbox)
        tracker.init(frame, bbox)
        trackers.append(tracker)

    face_locations = []
    for i in bboxes:
        x1, y1, x2, y2 = i
        y = (y1, x2, y2, x1)
        face_locations.append(y)
    rgb_frame = frame[:, :, ::-1]

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        diff = np.subtract(saved_embeds, face_encoding)
        dist = np.sum(np.square(diff), axis=1)
        idx = np.argmin(dist)

        if dist[idx] < 0.29:
            face_names.append(names[idx])
        else:
            face_names.append("unknown")

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "unknown":
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    reDetect = 0
    fps = 0.0

    while True:
        ret, frame = video_capture.read()
        start = time.time()
        reDetect = np.mod(reDetect+1, 20)
        if frame is not None:

            if reDetect == 0:
                trackers = []
                colors = []
                bboxes, labels, probs = detect(frame, ort_session, input_name)
                ##print(bboxes)
                for bbox in bboxes:
                    colors.append((randint(0, 255)))
                    tracker = cv2.TrackerKCF_create()
                    bbox = tuple(bbox)
                    tracker.init(frame, bbox)
                    trackers.append(tracker)

                face_locations = []
                for i in bboxes:
                    x1, y1, x2, y2 = i
                    y = (y1, x2, y2, x1)
                    face_locations.append(y)
                rgb_frame = frame[:, :, ::-1]

                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    diff = np.subtract(saved_embeds, face_encoding)
                    dist = np.sum(np.square(diff), axis=1)
                    idx = np.argmin(dist)

                    if dist[idx] < 0.29:
                        face_names.append(names[idx])
                    else:
                        face_names.append("unknown")

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    ##if name == "unknown":
                        ##continue

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            else:

                for i in range(len(trackers)):
                    tracker = trackers[i]
                    ok, bbox = tracker.update(frame)
                    bboxes[i] = list(bbox)

                face_locations = []
                for i in bboxes:
                    x1, y1, x2, y2 = i
                    y = (y1, x2, y2, x1)
                    face_locations.append(y)
                rgb_frame = frame[:, :, ::-1]
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    #if name == "unknown":
                        #continue

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

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
