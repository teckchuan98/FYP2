import unittest
import onnxruntime as ort
from Testing.tools import detect
import cv2


class FaceDetectionUnitTest(unittest.TestCase):

    def test_1(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_2(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_3(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_4(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_5(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_6(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 9)

    def test_7(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/7.png')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 3)

    def test_8(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 3)

    def test_9(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 20)

    def test_10(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/10.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)


    def test_11(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/11.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_12(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/AngleChange/12.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_13(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/1.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_14(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_15(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_16(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/4.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_17(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/5.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_18(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/6.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 2)

    def test_19(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Blurry/7.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 3)
    def test_20(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 35)

    def test_21(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 25)

    def test_22(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)

    def test_23(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 26)

    def test_24(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_25(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 32)

    def test_26(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/7.png')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 10)

    def test_27(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)

    def test_28(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Crowd/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 6)


    def test_29(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/1.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_30(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_31(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_32(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_33(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/5.png')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_34(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_35(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/7.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_36(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_37(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_38(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/10.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)


    def test_39(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/11.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_40(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/12.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_41(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/LowLight/13.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_42(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_43(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_44(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/3.jfif')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 2)

    def test_45(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_46(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 4)

    def test_47(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 32)

    def test_48(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/7.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 5)

    def test_49(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/SpecialCase/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 10)

    def test_50(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/1.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_51(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/2.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_52(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/3.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_53(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/4.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_54(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/5.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_55(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/6.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_56(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/7.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_57(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/8.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_58(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/9.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_59(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/10.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_60(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/11.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_61(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/12.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_62(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/13.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_63(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/14.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_64(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/15.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_65(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/16.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_66(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/17.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_67(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/18.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_68(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/19.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_69(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/20.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_70(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/21.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_71(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/22.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_72(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/23.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_73(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/24.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_74(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/25.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_75(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/26.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_76(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/27.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_77(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/28.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_78(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/29.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_79(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/30.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_80(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/31.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_81(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/32.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_82(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/33.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_83(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/34.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_84(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/35.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_85(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/36.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_86(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/37.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_87(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/38.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_88(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/39.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_89(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/40.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_90(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/41.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_91(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/42.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_92(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/43.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_93(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/44.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_94(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/45.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_95(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/46.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_96(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/47.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_97(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/48.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_98(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/49.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_99(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/50.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

    def test_100(self):
        ort_session = ort.InferenceSession('ultra_light_640.onnx')
        input_name = ort_session.get_inputs()[0].name
        frame = cv2.imread('face_detection_test_cases/Normal/51.jpg')
        if frame is not None:
            x, count = detect(frame, ort_session, input_name)
            self.assertEqual(count, 1)

if __name__ == "__main__":
    unittest.main(exit=False)