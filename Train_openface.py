from imutils import paths
import pickle
import os
import cv2
import onnxruntime as ort
from detector import detect



print("Going through database")
imagePaths = list(paths.list_images("dataset"))

embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
input_name = ort_session.get_inputs()[0].name

embeddings = []
names = []
images = []

total = 0
#os.mkdir("detected")

for (i, imagePath) in enumerate(imagePaths):
    print("processing image " + str(i+1) + "/" + str(len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    #image = face_recognition.load_image_file(imagePath)
    #face_encoding = face_recognition.face_encodings(image)

    image = cv2.imread(imagePath)

    boxes, labels, probs = detect(image, ort_session, input_name)
    x1, y1, x2, y2 = boxes[0]

    face = image[y1:y2, x1:x2]
    (fH, fW) = face.shape[:2]
    # ensure the face width and height are sufficiently large
    if fW < 20 or fH < 20:
        continue
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    images.append(vec.flatten())
    names.append(name)
    #if len(face_encoding) > 0:
        #images.append(face_encoding[0])
        #names.append(name)

# write the actual face recognition model to disk
with open("embeddings.pkl", "wb") as f:
    pickle.dump((images, names), f)
