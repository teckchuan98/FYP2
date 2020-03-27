from imutils import paths
import pickle
import os
import face_recognition


print("Going through database")
imagePaths = list(paths.list_images("dataset"))

embeddings = []
names = []
images = []

total = 0
#os.mkdir("detected")

for (i, imagePath) in enumerate(imagePaths):
    print("processing image " + str(i+1) + "/" + str(len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = face_recognition.load_image_file(imagePath)

    face_encoding = face_recognition.face_encodings(image)

    if len(face_encoding) > 0:
        images.append(face_encoding[0])
        names.append(name)

# write the actual face recognition model to disk
with open("embeddings.pkl", "wb") as f:
    pickle.dump((images, names), f)
