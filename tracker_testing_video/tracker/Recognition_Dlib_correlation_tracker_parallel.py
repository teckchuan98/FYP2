import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect
import time
import multiprocessing as mp
import dlib
global tracking

def filter_name(cur_names, cur_locs):
    names = []
    locations = []

    for i in range(len(cur_names)):
        name = cur_names[i]
        if name != "unknown":

            names.append(name)
            locations.append(cur_locs[i])

    cur_names = names
    cur_locs = locations

    return cur_names, cur_locs

def track(box, pre_frame, cur_frame):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    print("initalize tracker", box)
    t = dlib.correlation_tracker()
    y1, x2, y2, x1 = box
    box = (x1, y1, x2, y2)
    print(box[0], box[1], box[2], box[3])
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])

    rgb = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
    t.start_track(rgb, rect)

    t.update(rgb2)
    pos = t.get_position()

    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    del t

    return startY, endX, endY, startX


def main():

    ort_session = ort.InferenceSession('../../ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    video_capture = cv2.VideoCapture("chandler.mp4")
    with open("../../embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))
    fps = 0.0

    redetect = -1
    face_locations = []
    face_names = []
    pre_frame = None
    n_processors = 8
    pool = mp.Pool(processes=n_processors)

    while True:
        redetect = (redetect + 1) % 20
        ret, frame = video_capture.read()
       # frame = cv2.resize(frame, (640, 480))  #
        start = time.time()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if redetect == 0:
                boxes, labels, probs = detect(frame, ort_session, input_name)
                temp = []
                for i in boxes:
                    x1, y1, x2, y2 = i
                    y = (y1, x2, y2, x1)
                    temp.append(y)
                rgb_frame = frame[:, :, ::-1]
                face_locations = temp
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

                face_names, face_locations = filter_name(face_names, face_locations)

                for i in range(len(face_locations)):
                    y1, x2, y2, x1 = face_locations[i]
                    y = (x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, face_names[i], (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            else:
                results = []
                for box in face_locations:
                    output = pool.apply_async(track, [box, pre_frame, frame])
                    results.append(output.get())

                face_locations = results
                for i in range(len(face_locations)):
                    y1, x2, y2, x1 = face_locations[i]
                    y = (x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, face_names[i], (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            pre_frame = frame

            div = time.time()-start
            if div != 0:
                fps = (fps + (1. / div)) / 2

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
