import cv2
import sys
import time
import onnxruntime as ort
from random import randint
from detector import detect
import multiprocessing as mp
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


def tracker_update(bboxes, frame, pre_frame):
    results = []
    for bbox in bboxes:
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(pre_frame, bbox)
        ok, new_bbox = tracker.update(frame)
        results.append(new_bbox)
    return results

def rr_partition(target, n_processors):
    results = []
    for i in range(n_processors):
        results.append([])

    for i in range(len(target)):
        results[i%n_processors].append(target[i])

    return results

if __name__ == '__main__':
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
    bboxes = []
    pre_frame = frame
    n_processor = 8
    pool = mp.Pool(processes=n_processor)

    startTracking = False

    while True:
        ok, frame = video.read()
        start = time.time()
        if not ok:
            break

        if not startTracking:
            bboxes, labels, probs = detect(frame, ort_session, input_name)
            bboxes = list(map(tuple, bboxes))
            if len(bboxes) >= 4:

                startTracking = True

        else:
            results = []
            bboxes = rr_partition(bboxes, n_processor)
            for subset in bboxes:

                out = pool.apply_async(tracker_update, [subset, frame, pre_frame])
                results.append(out.get())

            bboxes = []
            for result in results:
                for bbox in result:
                    bboxes.append(bbox)
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        pre_frame = frame
        end = time.time()
        print("The time used for each frame is " + str(end-start) + "milli second")
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
        93
        if k == 27: break



