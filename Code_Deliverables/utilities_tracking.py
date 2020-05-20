import cv2
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

def tagUI(frame, face_locations, face_names, probability, track):
    """
    Description : This function draw the bounding box around the recognized face. Used for UI.
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
        if name in track:
        # Draw a bounding box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        # write name of person below bounding box
            cv2.putText(frame, name + " : " + x, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    return frame