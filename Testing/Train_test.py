from imutils import paths
import pickle
import os
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def encode(i, imagePath, imagePaths):
    print("processing image " + str(i + 1) + "/" + str(len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = face_recognition.load_image_file(imagePath)
    face_encoding = face_recognition.face_encodings(image)

    if len(face_encoding)>0:
        return name, face_encoding[0]
    else:
        return name, []


def savemodel(names, images):
    data = {"embeddings": images, "names": names}

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    #recognizer = SVC(kernel="rbf", probability=True)
    recognizer = SVC(C=10, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the names and encodings to disk
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((images, names), f)

    # write the actual face recognition model to disk
    f = open("recognizer.pkl", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("le.pkl", "wb")
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