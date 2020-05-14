import onnxruntime as ort
from Testing.utilities2 import detect, recognise, initialise, tag
import time
import cv2


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name
    ort_session, input_name, recognizer, le, (saved_embeds, names) = initialise()

    frame = cv2.imread('cases/f1.jpg')

    start = time.time()
    if frame is not None:
        _, temp = detect(frame, ort_session, input_name)
        face_locations, face_names, probability = recognise(temp, frame, recognizer, le, names, saved_embeds)
        frame = tag(frame, face_locations, face_names, probability)
    end = time.time()

    cv2.waitKey(0)
    cv2.imwrite('cases/f1_output.jpg', frame)
    print(end-start)


if __name__ == "__main__":
    main()