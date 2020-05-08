import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect
import time

def update(cur_names, pre_names, cur_locs, pre_locs, cur_prob, pre_prob):

    print("in update")

    names = []
    locations = []
    probability = []

    for i in range(len(pre_names)):
        name = pre_names[i]
        if name not in cur_names and name != "unknown":
            names.append(name)
            locations.append(pre_locs[i])
            probability.append(pre_prob[i])

    if len(names) == 0:
        return cur_names, cur_prob

    unknowns = []
    for i in range(len(cur_names)):
        if cur_names[i] == "unknown":
            unknowns.append((i, -1, -1))


    for n in range(len(names)):
        face = locations[n]
        min_dif = None
        min_id = -1
        for i in range(len(unknowns)):
            cur_id = unknowns[i][0]
            face1 = cur_locs[cur_id]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            ##print("check")
            ##print(dif)
            if dif <= 300:
                if (min_dif == None or min_dif > dif):
                    min_dif = dif
                    min_id = i
        if(min_id != -1):
            if unknowns[min_id][1] == -1 or min_dif < unknowns[min_id][1]:
                cur_id =  unknowns[min_id][0]
                unknowns[min_id]= (cur_id, min_dif, n)

    for i in range(len(unknowns)):
        if unknowns[i][1] != -1:
            cur_id = unknowns[i][0]
            cur_names[cur_id] = names[unknowns[i][2]]
            cur_prob[cur_id] = probability[unknowns[i][2]]

    return cur_names, cur_prob

def track(pre_faces, cur_faces, names, probability):
    results = []
    results_names = []
    results_prob = []
    for i in range(len(pre_faces)):
        results.append((-1, (-1, -1, -1, -1)))
        results_names.append(names[i])
        results_prob.append(probability[i])

    for n in range(len(cur_faces)):
        face = cur_faces[n]
        min_dif = None
        min_id = -1
        for i in range(len(pre_faces)):
            face1 = pre_faces[i]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            ##print("check")
            ##print(dif)
            if dif <= 300:
                if (min_dif == None or min_dif > dif):
                    min_dif = dif
                    min_id = i
        if(min_id != -1):
            if results[min_id][0] == -1 or min_dif < results[min_id][0]:
                results[min_id]= (min_dif, cur_faces[n])

    temp = results
    temp_names = results_names
    temp_prob = results_prob
    results_names = []
    results = []
    results_prob = []

    for i in range(len(temp)):
        result = temp[i]
        if result[0] != -1:
            results.append(result[1])
            results_names.append(temp_names[i])
            results_prob.append((temp_prob[i]))

    return results, results_names, results_prob

def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    recognizer = pickle.loads(open("recognizer.pkl", "rb").read())
    le = pickle.loads(open("le.pkl", "rb").read())

    video_capture = cv2.VideoCapture('chandler.mp4')
    #with open("embeddings.pkl", "rb") as f:
        #(saved_embeds, names) = pickle.load(f)

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))
    fps = 0.0
    redetect = -1
    face_locations = []
    face_names = []
    probability = []

    while True:
        redetect = (redetect+1)%20
        ret, frame = video_capture.read()
        start = time.time()
        # frame = cv2.resize(frame, (320, 240))

        if frame is not None:

            boxes, labels, probs = detect(frame, ort_session, input_name)
            temp = []
            for i in boxes:
                x1, y1, x2, y2 = i
                y = (y1, x2, y2, x1)
                temp.append(y)
            rgb_frame = frame[:, :, ::-1]

            if redetect == 0:
                ##print("redeteced")
                face_encodings = face_recognition.face_encodings(rgb_frame, temp)
                names = []
                cur_prob = []

                for face_encoding in face_encodings:

                    face_encoding = [face_encoding]
                    preds = recognizer.predict_proba(face_encoding)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    cur_prob.append(proba)
                    ##print(proba, name)
                    if proba > 0.7:
                        names.append(name)
                    else:
                        names.append("unknown")

                names, cur_prob = update(names, face_names, temp, face_locations, cur_prob, probability)

                face_locations = temp
                face_names = names
                probability = cur_prob

            else:
                ##print("updated")
                face_locations, face_names, probability = track(face_locations, temp, face_names, probability)

            ##print(face_locations, face_names, probability)
            for (top, right, bottom, left), name, prob in zip(face_locations, face_names, probability):
                if name == "unknown":
                   continue

                x = prob * 100
                x = str(x)
                x = x[:3]
                x = x + "%"

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name + " : " + x, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            fps = (fps + (1. / (time.time() - start))) / 2
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
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
