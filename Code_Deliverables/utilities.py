import cv2
import numpy as np
from Code_Deliverables.Compute_detections import predict
import face_recognition
import onnxruntime as ort
import pickle
import dlib
import time

def remove_unknown(cur_names, cur_loc, cur_prob):

    names = []
    locations = []
    probability = []

    for i in range(len(cur_names)):
        if cur_names[i] != "unknown":
            names.append(cur_names[i])
            locations.append(cur_loc[i])
            probability.append(cur_prob[i])

    return names, locations, probability

def track_by_tracker(box, pre_frame, cur_frame, id):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    print("initalize tracker", box)
    t = dlib.correlation_tracker()
    y1, x2, y2, x1 = box
    box = (x1, y1, x2, y2)
    print(box[0], box[1], box[2], box[3])
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])

    rgb = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
    start = time.time()
    t.start_track(rgb, rect)
    end = time.time()
    print("time taken for initialize: ", end-start, id)

    start = time.time()
    t.update(rgb2)
    end = time.time()
    print("time taken for update: ", end - start, id)
    pos = t.get_position()

    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    del t

    return startY, endX, endY, startX

def remove_duplicate(cur_names, cur_locs, cur_prob):
    names = []
    locations = []
    probability = []

    for i in range(len(cur_names)):
        name = cur_names[i]
        prob = cur_prob[i]
        max_id = i
        if name != "unknown":
            if not name in names:
                names.append(name)
                for j in range(len(cur_names)):
                    name2 = cur_names[j]
                    prob2 = cur_prob[j]
                    if name2 == name and prob2 > prob and name2 != "unknown":
                        max_id = j
                        prob = prob2

                probability.append(prob)
                locations.append(cur_locs[max_id])
        else:
            names.append(name)
            probability.append(prob)
            locations.append(cur_locs[max_id])

    cur_names = names
    cur_locs = locations
    cur_prob = probability

    return cur_names, cur_prob, cur_locs


def update(cur_names, pre_names, cur_locs, pre_locs, cur_prob, pre_prob, false_track):
    threshold = 2
    names = []
    locations = []
    probability = []

    unknowns = []
    for i in range(len(cur_names)):
        name = cur_names[i]
        if name == "unknown":
            unknowns.append((i, -1, -1))
        else:
            false_track[name] = 0

    for i in range(len(pre_names)):
        name = pre_names[i]
        if name not in cur_names and name != "unknown":
            if name not in false_track:
                false_track[name] = 0
            else:
                if false_track[name] > threshold:
                    false_track[name] = 0
                else:
                    false_track[name] += 1
                    names.append(name)
                    locations.append(pre_locs[i])
                    probability.append(pre_prob[i])

    if len(names) == 0:
        return cur_names, cur_prob, cur_locs, false_track

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
                if min_dif == None or min_dif > dif:
                    min_dif = dif
                    min_id = i
        if min_id != -1:
            if unknowns[min_id][1] == -1 or min_dif < unknowns[min_id][1]:
                cur_id = unknowns[min_id][0]
                unknowns[min_id] = (cur_id, min_dif, n)

    for i in range(len(unknowns)):
        if unknowns[i][1] != -1:
            cur_id = unknowns[i][0]
            cur_names[cur_id] = names[unknowns[i][2]]
            cur_prob[cur_id] = probability[unknowns[i][2]]

    return cur_names, cur_prob, cur_locs, false_track


def initialise():
    """
    Description : This function loads detection model, recognition model and video file
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    param : none
    Return :
            ort_seesion: The initialised face detector model
            recognizer: the initialised recognition model
            le: label encoder of recognition model
            (saved_embeds, names): saved embeddings and names from dataset
            video_capture: the video input from video file
            w: width of video frame
            h: height of video frame
            out: output file to write processed video
    """
    ort_session = ort.InferenceSession('models/ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    recognizer = pickle.loads(open("models/recognizer.pkl", "rb").read())
    le = pickle.loads(open("models/le.pkl", "rb").read())
    with open("models/embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)
    video_path = 'inputs/chandler'
    video_capture = cv2.VideoCapture(video_path + ".mp4")
    output_path = "outputs/" + video_path + "_output.mp4"

    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))

    return ort_session, input_name, recognizer, le, (saved_embeds, names), video_capture, w, h, out

def initialiseRecognizer():
    """
    Description : This function loads recognition model
    Author : Jeetun Ishan
    Last modified : 20/05/2020
    param : none
    Return :
            ort_seesion: The initialised face detector model
            recognizer: the initialised recognition model
            le: label encoder of recognition model
            (saved_embeds, names): saved embeddings and names from dataset
    """
    recognizer = pickle.loads(open("models/recognizer.pkl", "rb").read())
    le = pickle.loads(open("models/le.pkl", "rb").read())
    with open("models/embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    return recognizer, le, (saved_embeds, names)

def initialiseDetector():
    """
    Description : This function loads detection model
    Author : Jeetun Ishan
    Last modified : 20/05/2020
    param : none
    Return :
            ort_seesion: The initialised face detector model
    """
    ort_session = ort.InferenceSession('models/ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name

    return ort_session, input_name


def detect(frame, ort_session, input_name):
    """
    Description : This function takes the frame, ort session, input name and returns an rgb frame and a list of detected
                  face locations
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    params :
            frame: The video frame to perform detection
            ort-session: The onnx detection model
            input_name: input name of frame
    Return :
            rgb_frame: the frame converted to rgb format
            temp: List of detected faces in x1, y1, x2, y2 format
    Reference:
            https://github.com/fyr91/face_detection/blob/master/detect_ultra_light.py
    """

    #
    h, w, _ = frame.shape

    # processing image starts here
    # referenced from https://github.com/fyr91/face_detection/blob/master/detect_ultra_light.py
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # run the detection model on current frame
    confidences, boxes = ort_session.run(None, {input_name: img})

    # the predict function will predict all boxes with human faces by using an IOU threshold
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

    temp = []
    # changing the coordinates position as required for face recognition function
    for i in boxes:
        x1, y1, x2, y2 = i
        y = (y1, x2, y2, x1)
        temp.append(y)
    rgb_frame = frame[:, :, ::-1]

    return rgb_frame, temp


def recognise(temp, rgb_frame, recognizer, le, names, saved_embeds):
    """
    Description : This function takes the detected face locations and computes the embeddings and classify the face as
                    being recognised or not
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    param :
            temp: the detected face locations from the frame
            rgb_frame: the current frame in rgb format
            recognizer: recognition model
            le: label encoder
            names: names of person in dataset
            saved embeds: the embeddings of people present in dataset
    Return :
            face_locations: location of recognised faces in x1, y1, x2, y2 format
            face_names: names of recognised faces
            probability: probability of classification
    Reference:
            https://gist.github.com/fyr91/12fc19c26dbb82e9a019f0203397ae16#file-compare-py
    """
    face_locations = temp

    # face_encodings function will take the current frame and face locations in the frame and will compute the face
    # embeddings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    probability = []

    # Classification is performed for each face embedding generated and the euclidean distance of the embedding and
    # the embeddings from dataset is calculated

    for face_encoding in face_encodings:

        # euclidean distance process below

        diff = np.subtract(saved_embeds, face_encoding)
        dist = np.sum(np.square(diff), axis=1)
        idx = np.argmin(dist)

        # if shortest distance is less than 0.29
        if dist[idx] < 0.29:
            id = names[idx]
        else:
            id = "unknown"

        face_encoding = [face_encoding]

        # perform classification
        preds = recognizer.predict_proba(face_encoding)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # Conditions below ensures best recognition accuracy
        if proba > 0.6 and name == id:
            face_names.append(name)
            probability.append(proba)
        elif proba > 0.7 and id == "unknown" and name != "unknown":
            face_names.append(name)
            probability.append(proba)
        elif id != "unknown" and name == "unknown":
            face_names.append(id)
            probability.append(proba)
        else:
            face_names.append("unknown")
            probability.append(proba)

    return face_locations, face_names, probability


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
        if (min_id != -1):
            if results[min_id][0] == -1 or min_dif < results[min_id][0]:
                results[min_id] = (min_dif, cur_faces[n])

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


def tag(frame, face_locations, face_names, probability):
    """
    Description : This function draw the bounding box around the recognized face
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    param :
            frame: the frame to tag the person
            face_locations: list of locations of the recognized persons
            face_names: list of names of the recognised persons
            probability: list of probability of classification
    Return :
            frame: the frame with the recognised persons tagged

    """
    for (top, right, bottom, left), name, prob in zip(face_locations, face_names, probability):
        if name == "unknown":
            continue
        x = prob * 100
        x = str(x)
        x = x[:3]
        x = x + "%"
        # Draw a bounding box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        # write name of person below bounding box
        cv2.putText(frame, name + " : " + x, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    return frame
