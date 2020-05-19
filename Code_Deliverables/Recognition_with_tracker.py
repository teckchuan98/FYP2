import cv2
from IndividualRecogniton.detector import detect
import time
import dlib
import multiprocessing as mp
from Code_Deliverables.utilities import detect, recognise, tag, initialise, update, remove_duplicate, remove_unknown, \
    track_by_tracker


def main():
    ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out = initialise()
    fps = 0.0
    redetect = -1
    face_locations = []
    face_names = []
    probability = []

    pre_frame = None
    false_track = {}

    cur_id = 0
    n_processors = 8
    pool = mp.Pool(processes=n_processors)

    while True:
        redetect = (redetect + 1) % 15
        ret, frame = video_capture.read()
        if pre_frame is None:
            pre_frame = frame
        start = time.time()

        if frame is not None:

            rgb_frame, temp = detect(frame, ort_session, input_name)

            if redetect == 0:
                cur_names = []
                cur_prob = []
                temp, cur_names, cur_prob = recognise(temp, rgb_frame, recognizer, le, names, saved_embeds)
                cur_names, cur_prob, temp = remove_duplicate(cur_names, temp, cur_prob)
                cur_names, cur_prob, temp, false_track = update(cur_names, face_names, temp, face_locations, cur_prob, probability, false_track)
                face_locations = temp
                face_names = cur_names
                probability = cur_prob

            else:
                results = []
                face_names, face_locations, probability = remove_unknown(face_names, face_locations, probability)
                for i in range(len(face_locations)):
                    box = face_locations[i]
                    cur_id += 1
                    output = pool.apply_async(track_by_tracker, [box, pre_frame, frame, cur_id])
                    results.append(output.get())

                face_locations = results

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
