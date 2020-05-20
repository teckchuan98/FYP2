import cv2

def remove_duplicate(cur_names, cur_locs, cur_prob):
    """
    Description : This function select the faces names with highest probability, if there is multiple persons recognized as the same identity.
    Author : Tan Kai Yi
    Last modified : 20/05/2020
    param :
            cur_names: the current detected faces names, faces cant be recognized will be named "unknown"
            cur_locs: the current detected faces locations
            cur_prob: the current detected faces probabilities. The probability show the similarity of a person to the recognition result identity

    Return :
            names: the updated detected faces names, where the duplicate names is filtered out and only the name with highest probability is in the result list
            locations: the updated detected faces locations, where the location of faces with duplicate name is filtered out
            probability: the updated probability of detected faces, where the probabilities of faces with duplicate name is filtered out

    """
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

    return names, probability, locations


def update(cur_names, pre_names, cur_locs, pre_locs, cur_prob, pre_prob, false_track):
    """
    Description : This function is used to untrack the person who continuously fail to be recognized (false track) more than a specific times (threshold) in the recgnition process
    Author : Tan Kai Yi
    Last modified : 20/05/2020
    param :
            cur_names: the current detected faces names, faces cant be recognized will be named "unknown"
            pre_names: the previous detected faces names in previous frame, faces cant be recognized will be named "unknown"
            cur_locs: the current detected faces locations
            pre_locs: the previous detected faces locations in previous frame
            cur_prob: the current detected faces probabilities. The probability show the similarity of a person to the recognition result identity
            pre_prob: the previous detected faces probabilities in previous frame
            false_track: a dictionary to store the times of a person continuously fail to be recognized in the recognition process (false track)

    Return :
            cur_names: the updated detected faces names, where names that has a false track more than a specific times is removed
            cur_locs: the updated detected faces locations, where the location of faces that has a false track more than a specific times is removed
            cur_prob: the updated probability of detected faces, where the probabilities of faces that has a false track more than a specific times is removed
            false_track: the updated dictionary of the number of false track for each person. If a person has false track more than a specific time, it will be removed in the cur_names list and his/her false track count will reset to 0 in dictionary

    """
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


def track(pre_faces, cur_locs, cur_names, probability):
    """
    Description : This function to track person who is successfully recognized, as the recognition process will only run once per n frames. During this period, the person need to be tracked.
    Author : Tan Kai Yi
    Last modified : 20/05/2020
    param :
            cur_names: the current detected faces names, faces cant be recognized will be named "unknown"
            cur_locs: the current detected faces locations
            probability: the current detected faces probabilities. The probability show the similarity of a person to the recognition result identity

    Return :
            results: the updated detected faces locations
            results_names: the updated detected faces names, which aligned to results (results[i] is the face location with name results_names[i])
            results_prob: the updated probability of detected faces, which aligned to results (results[i] is the face location with probability results_prob[i])

    """
    results = []
    results_names = []
    results_prob = []
    for i in range(len(pre_faces)):
        results.append((-1, (-1, -1, -1, -1)))
        results_names.append(cur_names[i])
        results_prob.append(probability[i])

    for n in range(len(cur_locs)):
        face = cur_locs[n]
        min_dif = None
        min_id = -1
        for i in range(len(pre_faces)):
            face1 = pre_faces[i]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            if dif <= 300:
                if (min_dif == None or min_dif > dif):
                    min_dif = dif
                    min_id = i
        if (min_id != -1):
            if results[min_id][0] == -1 or min_dif < results[min_id][0]:
                results[min_id] = (min_dif, cur_locs[n])

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