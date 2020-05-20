import cv2
import time
from Code_Deliverables.utilities import detect, recognise, tag, initialise


def main():

    ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out = initialise()
    fps = 0.0

    while True:
        ret, frame = video_capture.read()
        start = time.time()

        if frame is not None:

            rgb_frame, temp = detect(frame, ort_session, input_name)
            face_locations, face_names, probability = recognise(temp, rgb_frame, recognizer, le, names, saved_embeds)
            print(face_names)
            frame = tag(frame, face_locations, face_names, probability)

            fps = (fps + (1. / (time.time() - start))) / 2
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
