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
from collections import Counter

def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name
    video_capture = cv2.VideoCapture('rdj.mp4')

    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
    threshold = 0.63

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
                        predictions = []

                        embeds = compute_embeddings(faces, images_placeholder, sess, phase_train_placeholder, embeddings)


                        for embedding in embeds:
                            diff = np.subtract(saved_embeds, embedding)
                            dist = np.sum(np.square(diff), axis=1)
                            #idx = np.argmin(dist)
                            rr_model = []
                            for calculated in dist:
                                if calculated < threshold:
                                    rr_model.append(True)
                                else:
                                    rr_model.append(False)
                            new = []
                            for r in range(len(rr_model)):
                                new.append((names[r], rr_model[r]))
                            # new = [ [robert,true], [robert,true], [tom,false] ]

                            true = []
                            for t in new:
                                if t[1]:
                                    true.append(t)
                            # true = [ (robert,true), (robert, true), (tom, true) ]
                            probability = 0
                            if len(true) > 0:
                                c = Counter(true)
                                c = c.most_common(1)  # [ ( (robert, true), 5 ) ]
                                tot = names.count(c[0][0][0])
                                probability = c[0][1]/tot

                            if probability > 0.1:
                                predictions.append(c[0][0][0])
                            else:
                                predictions.append("unknown")

                        # draw
                        frame = draw(boxes, predictions, frame)

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    main()
