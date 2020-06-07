import cv2
import time
from Unit_Testing_Performance.utilities2 import detect, recognise, track, tag, update, remove_duplicate,initialise_video_test
import dlib
import multiprocessing.dummy as mp

def main():
    result_file, ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out = initialise_video_test()
    fps = 0.0

    redetect = -1
    face_locations = []
    face_names = []
    probability = []
    pre_frame = None
    false_track = {}
    redetect_threshold = 0
    redetect_freqeunt = 15
    frame_count = 0
    fps_avr = 0
    result_per_sec = set()
    n_processors = 8
    trackers = []
    pool = mp.Pool(processes=n_processors)

    while True:
        redetect = (redetect + 1) % redetect_freqeunt
        ret, frame = video_capture.read()
        if pre_frame is None:
            pre_frame = frame
        start = time.time()

        if frame is not None:

            rgb_frame, temp = detect(frame, ort_session, input_name)

            if redetect == 0 or len([a for a in face_names if a != "unknown"]) <= redetect_threshold:
                temp, cur_names, cur_prob = recognise(temp, rgb_frame, recognizer, le, names, saved_embeds)
                print(cur_names)
                cur_names, cur_prob, temp = remove_duplicate(cur_names, temp, cur_prob)
                cur_names, cur_prob, temp, false_track = update(cur_names, face_names, temp, face_locations, cur_prob, probability, false_track)
                face_locations = temp
                face_names = cur_names
                probability = cur_prob
                del(trackers)
                trackers = []
                for i in range(len(face_locations)):
                    top, right, bottom, left = face_locations[i]
                    box = (left, top, right, bottom)
                    # construct a dlib rectangle object from the bounding box
                    # coordinates and then start the correlation tracker

                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                    t.start_track(rgb_frame, rect)
                    trackers.append(t)

            else:

                results = []
                for tracker in trackers:
                    output = pool.apply_async(track, [tracker,frame])
                    results.append(output.get())
                face_locations = results

            frame = tag(frame, face_locations, face_names, probability)

            fps = (1. / (time.time() - start))
            fps_avr += fps
            frame_count += 1
            if frame_count % 30 != 0 :
                for name in face_names:
                    if name != "unknown":
                         result_per_sec.add(name)

            elif frame_count != 0 and frame_count %30 == 0:
                for name in face_names:
                    if name != "unknown":
                         result_per_sec.add(name)
                period = frame_count/30
                period_txt = str(period-1) + ".00 - " + str(period) + ".00 : "
                for name in result_per_sec:
                    period_txt += " " + name
                period_txt += "\n"
                result_per_sec.clear()
                result_file.write(period_txt)

            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            result_file
            out.write(frame)

            pre_frame = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                result_file.close()
                break

        else:
            print(fps_avr/frame_count)
            result_file.close
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()