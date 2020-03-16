# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-30 11:25:26
import cv2
import onnxruntime as ort
from Face_detector import detect

def main():
    video_capture = cv2.VideoCapture('test.mp4')
    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w),int(h)))

    while True:
        ret, frame = video_capture.read()

        if frame is not None:
            detected_image = detect(frame, ort_session, input_name)
            # out.write(detected_image)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
