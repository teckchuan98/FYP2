import cv2
import time
from Code_Deliverables.utilities import detect, recognise, track, tag, initialise, update, remove_duplicate


def main():
    ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out = initialise()
    fps = 0.0

    redetect = -1
    face_locations = []
    face_names = []
    probability = []
    pre_frame = None
    false_track = {}
    redetect_threshold = 0
    redetect_freqeunt = 15

    while True:
        redetect = (redetect + 1) % redetect_freqeunt
        ret, frame = video_capture.read()
        if pre_frame is None:
            pre_frame = frame
        start = time.time()

        if frame is not None:

            rgb_frame, temp = detect(frame, ort_session, input_name)

            if redetect == 0 or len([a for a in face_names if a != "unknown"]) <= redetect_threshold:
                cur_names = []
                cur_prob = []
                temp, cur_names, cur_prob = recognise(temp, rgb_frame, recognizer, le, names, saved_embeds)
                print(cur_names)
                cur_names, cur_prob, temp = remove_duplicate(cur_names, temp, cur_prob)
                cur_names, cur_prob, temp, false_track = update(cur_names, face_names, temp, face_locations, cur_prob, probability, false_track)
                face_locations = temp
                face_names = cur_names
                probability = cur_prob

            else:
                face_locations, face_names, probability = track(face_locations, temp, face_names, probability)

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