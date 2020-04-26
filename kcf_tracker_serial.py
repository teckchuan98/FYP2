import cv2
import sys
import onnxruntime as ort
from random import randint
from detector import detect
import time
import numpy as np

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

'''
def update_tracker(trackers, frame):

    for tracker in trackers:
        tracker.update(frame)

def rr_partition(trackers, n_processor):
    results = []
    for i in range(n_processor):
        results.append([])

    for i in range(len(trackers)):
        results[i%n_processor].append(trackers[i])

    return results
    
'''

def tracker_update(tracker, frame):
    ok, bbox = tracker.update(frame)
    return ok, bbox

if __name__ == '__main__' :
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name
    video = cv2.VideoCapture('zoom_one_face.mp4')

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
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(frame, bbox)
        trackers.append(tracker)

    reDetect = 0

    while True:
        ok, frame = video.read()
        start = time.time()
        if not ok:
            break

        if len(trackers) < 1:
            trackers = []
            bboxes, labels, probs = detect(frame, ort_session, input_name)
            bboxes = list(map(tuple, bboxes))
            for bbox in bboxes:
                colors.append((randint(0, 255)))
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)
                trackers.append(tracker)

        for tracker in trackers:
            ok, bbox = tracker_update(tracker, frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        end = time.time()
        print("The time used for each frame is " + str(end - start) + "milli second")
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
        93
        if k == 27: break



