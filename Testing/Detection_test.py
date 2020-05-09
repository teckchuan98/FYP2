import onnxruntime as ort
from Testing.tools import detect
import time
import cv2


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    frame = cv2.imread('test2.jpg')

    start = time.time()
    if frame is not None:
        detect(frame, ort_session, input_name)
    end = time.time()

    cv2.waitKey(0)
    print(end-start)


if __name__ == "__main__":
    main()