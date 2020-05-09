import threading

import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect
import time
import multiprocessing
import dlib
global tracking
tracking = True

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

def start_tracker(box, label, rgb, inputQueue, outputQueue, id):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    print("initalize tracker", box)
    t = dlib.correlation_tracker()
    print(box[0], box[1], box[2], box[3])
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)

    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while tracking:
        print(id)
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()

        # if there was an entry in our queue, process it
        if rgb is not None and tracking:
            # update the tracker and grab the position of the tracked
            # object
            t.update(rgb)
            pos = t.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the label + bounding box coordinates to the output
            # queue
            outputQueue.put((label, (startX, startY, endX, endY)))

    del t
    print(t)
    print("deleted")
    return None

def main():

    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    video_capture = cv2.VideoCapture("chandler.mp4")
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))
    fps = 0.0

    redetect = -1
    face_locations = []
    face_names = []

    # initialize our list of queues -- both input queue and output queue
    # for *every* object that we will be tracking
    cur_id = 0
    p = None

    while True:
        redetect = (redetect + 1) % 20
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (640, 480))  #
        start = time.time()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if redetect == 0:
                tracking = False
                cur_id += 1
                inputQueues = []
                outputQueues = []
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
                    # create two brand new input and output queues,
                    # respectively
                    iq = multiprocessing.Queue()
                    oq = multiprocessing.Queue()
                    inputQueues.append(iq)
                    outputQueues.append(oq)

                    # spawn a daemon process for a new object tracker
                    print(y)
                    p = multiprocessing.Process(
                        target=start_tracker,
                        args=(y, face_names[i], rgb, iq, oq, cur_id))
                    p.daemon = True
                    tracking = True
                    p.start()

            else:
                # loop over each of our input ques and add the input RGB
                # frame to it, enabling us to update each of the respective
                # object trackers running in separate processes
                for iq in inputQueues:
                    iq.put(rgb)

                # loop over each of the output queues
                for oq in outputQueues:
                    # grab the updated bounding box coordinates for the
                    # object -- the .get method is a blocking operation so
                    # this will pause our execution until the respective
                    # process finishes the tracking update
                    (label, (startX, startY, endX, endY)) = oq.get()
                    print(label, startX, startY, endX, endY)
                    # draw the bounding box from the correlation object
                    # tracker
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

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
