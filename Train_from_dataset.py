#train svm from dataset
#export svm via pickle

from imutils import paths
import numpy as np
import pickle
import cv2
import os
import onnxruntime as ort
from Compute_detections import predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ort_session = ort.InferenceSession('ultra_light_640.onnx')
input_name = ort_session.get_inputs()[0].name
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

print("Going through database")
imagePaths = list(paths.list_images("dataset"))

embeddings = []
names = []

total = 0

dirname = 'detected faces'
os.mkdir(dirname)

for (i, imagePath) in enumerate(imagePaths):
    print("processing image " + str(i+1) + "/" + str(len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    h, w, _ = image.shape

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

    box = boxes[0]
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]

    (height, width) = crop.shape[:2]
    if width < 20 or height < 20:
        continue

    blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob)
    vec = embedder.forward()

    names.append(name)
    embeddings.append(vec.flatten())
    total += 1

    #cv2.imwrite(os.path.join(dirname, name + "_" + str(i + 1) + ".jpg"), crop)

data = {"embeddings": embeddings, "names": names}
#f = open("embeddings.pickle", "wb")
#f.write(pickle.dumps(data))
#f.close()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["names"])

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("label_encoder.pickle", "wb")
f.write(pickle.dumps(label_encoder))
f.close()

f = open("recognizer_model.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()








