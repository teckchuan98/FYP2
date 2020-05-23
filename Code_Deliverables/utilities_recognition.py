import numpy as np
import face_recognition
import pickle


def initialiseRecognizer():
    """
    Description : This function loads recognition model
    Author : Jeetun Ishan
    Last modified : 20/05/2020
    param : none
    Return :
            recognizer: the initialised recognition model
            le: label encoder of recognition model
            (saved_embeds, names): saved embeddings and names from dataset
    """
    recognizer = pickle.loads(open("models/recognizer.pkl", "rb").read())
    le = pickle.loads(open("models/le.pkl", "rb").read())
    with open("models/embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    return recognizer, le, (saved_embeds, names)


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
        if proba > 0.8:
            face_names.append(name)
            probability.append(proba)
        elif name == id and proba > 0.7:
            face_names.append(name)
            probability.append(proba)
        else:
            face_names.append("unknown")
            probability.append(proba)

    return face_locations, face_names, probability
