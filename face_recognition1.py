import numpy as np
import pickle
import cv2
from detector import detect
from detector import align
from detector import compute_embeddings
from detector import draw
import onnxruntime as ort
import tensorflow as tf
import dlib
from imutils import face_utils


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name
    video_capture = cv2.VideoCapture('chandler.mp4')

    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
    threshold = 0.6

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("recognizer.pkl", "rb").read())
    le = pickle.loads(open("label_encoder.pkl", "rb").read())

    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('models/mfn.ckpt.meta')
            saver.restore(sess, 'models/mfn.ckpt')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            while True:
                ret, frame = video_capture.read()
                if frame is not None:
                    boxes, labels, probs = detect(frame, ort_session, input_name)
                    faces = []

                    for i in range(boxes.shape[0]):
                        box = boxes[i]
                        aligned_face = align(box, frame, fa)
                        faces.append(aligned_face)

                    if len(faces) > 0:
                        embeds = compute_embeddings(faces, images_placeholder, sess, phase_train_placeholder, embeddings)
                        predictions = []
                        for embedding in embeds:
                            embedding = embedding.reshape(1, -1)
                            preds = recognizer.predict_proba(embedding)[0]
                            j = np.argmax(preds)
                            proba = preds[j]
                            name = le.classes_[j]

                            if proba > 0.8:
                                predictions.append(name)
                            else:
                                predictions.append("unknown")

                            #diff = np.subtract(saved_embeds, embedding)
                            #dist = np.sum(np.square(diff), axis=1)
                            #idx = np.argmin(dist)

                            #if dist[idx] < threshold:
                                #predictions.append(names[idx])
                            #else:
                                #predictions.append("unknown")


                        # draw
                        frame = draw(boxes, predictions, frame)

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    main()
