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
import multiprocessing as mp

def tracker_update(bboxes, frame, pre_frame):
    results = []
    for bbox in bboxes:
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(pre_frame, tuple(bbox))
        ok, new_bbox = tracker.update(frame)
        print("new bbox")
        print(new_bbox)
        results.append(new_bbox)
    return results

def rr_partition(target, n_processors):
    results = []
    for i in range(n_processors):
        results.append([])

    for i in range(len(target)):
        results[i%n_processors].append(target[i])

    print("partition result")
    print(results)
    return results

def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    video_capture = cv2.VideoCapture("zoom.mp4")
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))
    fps = 0.0

    ok, frame = video_capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    trackers = []
    colors = []
    bboxes = []

    reDetect = -1

    pre_frame = frame
    n_processor = 8
    pool = mp.Pool(processes=n_processor)
    bboxes = []
    partitioned = False

    while True:
        ret, frame = video_capture.read()
        start = time.time()
        reDetect = np.mod(reDetect+1, 20)

        if reDetect == 0 or len(bboxes) == 0:
            bboxes, labels, probs = detect(frame, ort_session, input_name)
            bboxes = list(map(tuple, bboxes))
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
                print(left, right, top, bottom)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)




        else:

            results = []

            print("before")
            print(bboxes)
            bboxes = rr_partition(bboxes, n_processor)
            print(bboxes)

            for subset in bboxes:
                out = pool.apply_async(tracker_update, [subset, frame, pre_frame])
                results.append(out.get())

            bboxes = []
            for result in results:
                for bbox in result:
                    bboxes.append(bbox)

            print("here")
            print(bboxes)

            face_locations = []
            for i in bboxes:
                print(i[0])
                x1, y1, x2, y2 = i
                y = (y1, x2, y2, x1)
                face_locations.append(y)

            print(face_locations)

            rgb_frame = frame[:, :, ::-1]
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                #if name == "unknown":
                    #continue

                # Draw a box around the face
                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        pre_frame = frame
        fps = (fps + (1. / (time.time() - start))) / 2
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)
        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
