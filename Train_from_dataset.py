import dlib
from imutils import paths
import numpy as np
import pickle
import cv2
import os
import onnxruntime as ort
from Compute_detections import predict
from imutils import face_utils
import tensorflow as tf


ort_session = ort.InferenceSession('ultra_light_640.onnx')
input_name = ort_session.get_inputs()[0].name
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))


print("Going through database")
imagePaths = list(paths.list_images("dataset"))

embeddings = []
names = []
images = []

total = 0

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # convert to gray

    # align and resize
    aligned_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
    aligned_face = cv2.resize(aligned_face, (112, 112))

    aligned_face = aligned_face - 127.5
    aligned_face = aligned_face * 0.0078125
    images.append(aligned_face)
    names.append(name)

with tf.Graph().as_default():
    with tf.Session() as sess:
        print("loading checkpoint ...")
        saver = tf.train.import_meta_graph('models/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn.ckpt')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        embeds = sess.run(embeddings, feed_dict=feed_dict)
        with open("embeddings.pkl", "wb") as f:
            pickle.dump((embeds, names), f)
        print("Done!")








