import cv2
import time
from modular.utilities import detect, recognise, track, tag, initialise

def main():
    ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out = initialise()
    fps = 0.0
    redetect = -1
    face_locations = []
    face_names = []
    probability = []
    pre_frame = None

    while True:
        redetect = (redetect + 1) % 15
        ret, frame = video_capture.read()
        if pre_frame is None:
            pre_frame = frame
        start = time.time()

        if frame is not None:

            rgb_frame, temp = detect(frame, ort_session, input_name)

            if redetect == 0:
                face_locations, face_names, probability = recognise(temp, rgb_frame, recognizer, le, names, saved_embeds)
            else:
                face_locations, face_names = track(face_locations, temp, face_names, frame, pre_frame)

            frame = tag(frame, face_locations, face_names, probability)

            fps = (fps + (1. / (time.time() - start))) / 2
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            out.write(frame)

            pre_frame = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()