import cv2
import sys
import onnxruntime as ort
from random import randint
from detector import detect
import numpy as np

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

if __name__ == '__main__' :
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name
    video = cv2.VideoCapture('chandler.mp4')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    trackers = []
    colors = []
    bboxes, labels, probs = detect(frame, ort_session, input_name)
    bboxes = list(map(tuple, bboxes))
    print(bboxes)
    for bbox in bboxes:
        colors.append((randint(0, 255)))
        tracker = cv2.TrackerMOSSE_create()
        ok = tracker.init(frame, bbox)
        trackers.append(tracker)

    reDetect = 0

    while True:
        reDetect = (np.mod(reDetect+1, 10))
        print(reDetect)
        ok, frame = video.read()
        if not ok:
            break

        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            print("Updated " + str(bbox))

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        if reDetect == 0:
            print("re-detect")
            bboxes.clear()
            bboxes, labels, probs = detect(frame, ort_session, input_name)
            bboxes = list(map(tuple, bboxes))
            print(bboxes)
            trackers.clear()
            colors.clear()
            for bbox in bboxes:
                colors.append((randint(0, 255)))
                tracker = cv2.TrackerMOSSE_create()
                ok = tracker.init(frame, bbox)
                trackers.append(tracker)

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        93
        if k == 27: break



