import cv2

def remove_duplicate(cur_names, cur_locs, cur_prob):
    """
    Description : This function select only one of the duplicate faces names with highest probability, if there is multiple persons recognized as the same identity.
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
        ## only compare the probability of duplicate name which is not unknown
        if name != "unknown":
            if not name in names:
                names.append(name)
                ## loop through the current face names, check if there is any duplication
                for j in range(len(cur_names)):
                    name2 = cur_names[j]
                    prob2 = cur_prob[j]
                    ## if a duplication is found, compare the probability and store the index of name with highest probability
                    if name2 == name and prob2 > prob and name2 != "unknown":
                        max_id = j
                        prob = prob2

                probability.append(prob)
                locations.append(cur_locs[max_id])
        ## if a name is unknown, just keep it, it might use for future update in track function
        else:
            names.append(name)
            probability.append(prob)
            locations.append(cur_locs[max_id])

    return names, probability, locations


def update(cur_names, pre_names, cur_locs, pre_locs, cur_prob, pre_prob, false_track):
    """
    Description : This function is used to untrack the person who continuously fail to be recognized (false track) more than a specific times (threshold) in the recgnition process.
                    Meanwhile, if a person is not recognized in current frame but successfully recognized or tracked in previous frame, it will continue to track the person until the person has false track more than specific time
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
    ## set a threshold, to decide untrack a person after how many times of false track
    threshold = 2
    ## the names array only store the name with false track less than threshold
    names = []
    locations = []
    probability = []

    unknowns = []
    # the cur_names is the current recognized faces. Once a face is successfully recognized, its false track will reset to 0. The dictionary store the number of false track "continuously" of a person
    for i in range(len(cur_names)):
        name = cur_names[i]
        if name == "unknown":
            unknowns.append((i, -1, -1))
        else:
            false_track[name] = 0

    ## process to count false track
    for i in range(len(pre_names)):
        name = pre_names[i]
        ## only check names is not in cur_names. If a name is in cur_names, means it is successfully recognized. There is no need to count false track
        if name not in cur_names and name != "unknown":
            ## if the name is first time has a false track, add it to dictionary with value 0
            if name not in false_track:
                false_track[name] = 0
            else:
                ## if a name have a number of false track larger than threshold, reset the count back to 0
                if false_track[name] > threshold:
                    false_track[name] = 0
                ## otherwise, add the name to names array, add its location to locations array, and add its probability to probability array
                else:
                    false_track[name] += 1
                    names.append(name)
                    locations.append(pre_locs[i])
                    probability.append(pre_prob[i])

    ## if names is empty, it means all previous detected faces is either recognized or have a false track more than threshold in current frame. No further update required.
    if len(names) == 0:
        return cur_names, cur_prob, cur_locs, false_track

    ## process to update the person recognized as "unknown", but successfully recognized or tracked in previous frame with false track less than threshold
    for n in range(len(names)):
        face = locations[n]
        min_dif = None
        min_id = -1
        for i in range(len(unknowns)):
            cur_id = unknowns[i][0]
            face1 = cur_locs[cur_id]
            dif = abs(face1[0] - face[0]) + abs(face1[1] - face[1]) + abs(face1[2] - face[2]) + abs(face1[3] - face[3])
            ## a threshold to check if there is any recognized faces in previous frame is close to the current detected but not recognized face location. If there is multiple, choose the closest one
            if dif <= 300:
                if min_dif == None or min_dif > dif:
                    min_dif = dif
                    min_id = i
        if min_id != -1:
            if unknowns[min_id][1] == -1 or min_dif < unknowns[min_id][1]:
                cur_id = unknowns[min_id][0]
                unknowns[min_id] = (cur_id, min_dif, n)

    for i in range(len(unknowns)):
        ## if any unknown person found a recognized face's location in previous frame which is close to him, update his/her name and probability to that face
        if unknowns[i][1] != -1:
            cur_id = unknowns[i][0]
            cur_names[cur_id] = names[unknowns[i][2]]
            cur_prob[cur_id] = probability[unknowns[i][2]]

    return cur_names, cur_prob, cur_locs, false_track

def track(tracker, cur_frame):
    """
    Description : Function to track person who is successfully recognized in previous frame, using the dlib correlation tracker.
    Author : Tan Kai Yi
    Last modified : 08/06/2020
    param :
            tracker: the dlib tracker used to track the face
            cur_frame: the current frame for the tracker to update its pre-track face location

    Return :
            startY, endX, endY, startX: the updated faces locations tracked by the tracker

    """
    rgb2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
    tracker.update(rgb2)
    pos = tracker.get_position()

    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    return startY, endX, endY, startX

def remove_unknown(cur_loc, cur_name, cur_prob):
    """
    Description : Function to remove the face with face name "unknown", and remove its face location and probability as well
    Author : Tan Kai Yi
    Last modified : 08/06/2020
    param :
            cur_loc: the current detected faces locations
            cur_name: the current detected faces names, faces cant be recognized will be named "unknown"
            cur_prob: the current detected faces probabilities. The probability show the similarity of a person to the recognition result identity

    Return :
            result_loc: the detected faces locations after remove unknown faces
            result_name: the detected faces names after remove names with "unknown"
            result_prob: the detected faces probabilities after remove unknown faces

    """
    result_loc = []
    result_name = []
    result_prob = []
    for i in range(len(cur_loc)):
        name = cur_name[i]
        if name != "unknown":
            result_loc.append(cur_loc[i])
            result_name.append(cur_name[i])
            result_prob.append(cur_prob[i])

    return result_loc, result_name, result_prob

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