import unittest
import onnxruntime as ort
from Testing.tools import detect
import cv2


class testdetect(unittest.TestCase):

    def test_1(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/1.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_2(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_3(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_4(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_5(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/5.png')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_6(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_7(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/7.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_8(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_9(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_10(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/10.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)


    def test_11(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/11.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_12(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/12.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_13(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('cases/LowLight/13.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main(exit=False)