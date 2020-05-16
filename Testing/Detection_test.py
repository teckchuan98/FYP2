import onnxruntime as ort
from Testing.tools import detect
import time
import cv2


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name
    frame = cv2.imread('cases/Crowd/6.jpg')
    if frame is not None:
        x, count = detect(frame, ort_session, input_name)

    cv2.waitKey(0)
    cv2.imwrite('cases/59_output.jpg', x)

if __name__ == "__main__":
    main()