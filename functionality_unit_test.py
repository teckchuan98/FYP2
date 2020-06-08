import os
import unittest
import onnxruntime as ort
from imutils import paths
import dlib

from utillities_detection import detect,initialiseDetector
from utilities_recognition import recognise, initialiseRecognizer
from utilities_tracking import track, update, remove_duplicate, remove_unknown
import cv2
import pickle

class FaceDetectionUnitTest(unittest.TestCase):

    def test_initialiseDetector(self):
        ort_session = ort.InferenceSession('../FYP2 git/models/ultra_light_640.onnx')
        test_ort_session, input_name = initialiseDetector()
        self.assertEqual(type(test_ort_session), type(ort_session))

    def test_detect(self):
        ort_session = ort.InferenceSession('../FYP2 git/models/ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('../FYP2 git/Unit_Testing_Performance/face_detection_test_cases/Normal/48.jpg')
        count = 0
        frame, cur_loc = detect(frame, ort_session, input_name)
        self.assertLess(0, len(cur_loc))

    def test_initialiseRecognizer(self):
        recognizer = pickle.loads(open("../FYP2 git/models/recognizer.pkl", "rb").read())
        test_recognizer, le, (saved_embeds, names) = initialiseRecognizer()
        self.assertEqual(type(test_recognizer), type(recognizer))

    def test_recognize(self):
        ort_session = ort.InferenceSession('../FYP2 git/models/ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        recognizer, le, (saved_embeds, names) = initialiseRecognizer()
        imagePaths = list(paths.list_images("dataset"))
        for (i, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            frame = cv2.imread(imagePath)
            if frame is not None:
                _, temp = detect(frame, ort_session, input_name)
                face_names = []
                face_locations, face_names, probability = recognise(temp, frame, recognizer, le, names, saved_embeds)
                if(len(temp) != 0):
                    self.assertLess(0, len(face_names))
                    break

    def test_track(self):
        pre_locs =  [(2, 2, 2, 2), (802, 802, 802, 802), (402, 402, 402, 402), (3001, 3001, 3001, 3001)]
        pre_names = ["Kai Yi", "Ishan", "Teck Chuan", "Jiawen"]
        pre_probability = [0.8, 0.7, 0.9, 0.6]
        frame = cv2.imread('../FYP2 git/Unit_Testing_Performance/face_detection_test_cases/Normal/48.jpg')
        rgb_frame = cv2.cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        trackers = []
        for i in range(len(pre_locs)):
            top, right, bottom, left = pre_locs[i]
            box = (left, top, right, bottom)
            # construct a dlib rectangle object from the bounding box
            # coordinates and then start the correlation tracker

            t = dlib.correlation_tracker()
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
            t.start_track(rgb_frame, rect)
            trackers.append(t)

        for tracker in trackers:
           loc = track(tracker, rgb_frame)
           self.assertEqual(len(loc), 4)


    def test_update(self):
        cur_names = ["Kai Yi", "unknown", "Teck Chuan", "unknown"]
        pre_names = ["Kai Yi", "Ishan", "Teck Chuan", "Jiawen"]
        cur_locs =  [(2, 2, 2, 2), (11, 11, 11, 11), (101, 101, 101, 101), (1001, 1001, 1001, 1001)]
        pre_locs = [(1, 1, 1, 1), (10, 10, 10, 10), (100, 100, 100, 100), (1000, 1000, 1000, 1000)]
        cur_prob = [0.8, 0.3, 0.6, 0.2]
        pre_prob = [0.9, 0.8, 0.7, 0.6]
        false_track = {
            "Jiawen" : 3,
            "Ishan" : 1
        }
        cur_names, cur_prob, cur_locs, false_track = update(cur_names, pre_names, cur_locs, pre_locs, cur_prob, pre_prob, false_track)
        self.assertEqual(cur_locs, [(2, 2, 2, 2), (11, 11, 11, 11), (101, 101, 101, 101), (1001, 1001, 1001, 1001)])
        self.assertEqual(cur_names,  ["Kai Yi", "Ishan", "Teck Chuan", "unknown"])
        self.assertEqual(cur_prob, [0.8, 0.8, 0.6, 0.2])
        self.assertEqual(false_track['Jiawen'], 0)
        self.assertEqual(false_track['Ishan'], 2)

    def test_remove_duplicate(self):
        cur_name = ["Kai Yi", "Kai Yi", "Kai Yi", "Kai Yi"]
        cur_loc = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4)]
        cur_prob = [0.5, 0.6, 0.9, 0.4]
        cur_name, cur_prob, cur_loc = remove_duplicate(cur_name, cur_loc, cur_prob)
        self.assertEqual(1, len(cur_name))
        self.assertEqual(1, len(cur_loc))
        self.assertEqual(1, len(cur_prob))
        self.assertEqual(cur_name[0], "Kai Yi")
        self.assertEqual(cur_loc[0], (3, 3, 3, 3))
        self.assertEqual(cur_prob[0], 0.9)

    def test_remove_unknown(self):
        cur_name = ["unknown", "Kai Yi", "unknown"]
        cur_loc = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)]
        cur_prob = [0.5, 0.6, 0.9]
        cur_loc, cur_name, cur_prob = remove_unknown(cur_loc, cur_name, cur_prob)
        self.assertEqual(1, len(cur_name))
        self.assertEqual(1, len(cur_loc))
        self.assertEqual(1, len(cur_prob))
        self.assertEqual(cur_name[0], "Kai Yi")
        self.assertEqual(cur_loc[0], (2, 2, 2, 2))
        self.assertEqual(cur_prob[0], 0.6)

if __name__ == "__main__":
    unittest.main(exit=False)