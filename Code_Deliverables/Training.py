from imutils import paths
import pickle
import os
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def encode(i, imagePath, imagePaths):
    """
    Description : This functions encodes the face in a frame
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    params :
            i: current frame number
            imagePath: current image file
            imagePaths: current location
    Return :
            name: name of person
            list: list of face encoding or empty list
    """
    print("processing image " + str(i + 1) + "/" + str(len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = face_recognition.load_image_file(imagePath)
    face_encoding = face_recognition.face_encodings(image)

    if len(face_encoding)>0:
        return name, face_encoding[0]
    else:
        return name, []


def savemodel(names, images):
    """
    Description : This functions trains svm model, saves the recognition model, label encoder and embeddings
    Author : Jeetun Ishan
    Last modified : 17/05/2020
    params :
            names: names of person
            images: embeddings of persons in dataset
    Return : none
    Reference: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
    """
    data = {"embeddings": images, "names": names}

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1000, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the names and encodings to disk
    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump((images, names), f)

    # write the actual face recognition model to disk
    f = open("models/recognizer.pkl", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("models/le.pkl", "wb")
    f.write(pickle.dumps(le))
    f.close()


def main():
    print("Going through database")
    imagePaths = list(paths.list_images("dataset"))
    names = []
    images = []
    for (i, imagePath) in enumerate(imagePaths):
        name, encoding = encode(i, imagePath, imagePaths)
        if len(encoding) > 0:
            images.append(encoding)
            names.append(name)
    savemodel(names, images)



if __name__ == "__main__":
    main()