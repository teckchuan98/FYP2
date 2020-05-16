import unittest
import onnxruntime as ort
from Testing.tools import detect
import cv2


class testdetect(unittest.TestCase):

    def test_1(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 35)

    def test_2(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 25)

    def test_3(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)

    def test_4(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 26)

    def test_5(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_6(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 32)

    def test_7(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/7.png')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 10)

    def test_8(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)

    def test_9(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/Crowd/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)


if __name__ == "__main__":
    unittest.main(exit=False)