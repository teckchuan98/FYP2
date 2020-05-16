import unittest
import onnxruntime as ort
from Testing.tools import detect
import cv2


class testdetect(unittest.TestCase):

    def test_1(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_2(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_3(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/3.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 2)

    def test_4(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_5(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 4)

    def test_6(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 32)

    def test_7(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/7.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_8(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/SpecialCase/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 10)


if __name__ == "__main__":
    unittest.main(exit=False)