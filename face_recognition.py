import numpy as np
import pickle
import cv2
from Compute_detections import predict
import onnxruntime as ort
import tensorflow as tf
import dlib
from imutils import face_utils


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    video_capture = cv2.VideoCapture('test.mp4')

    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

    threshold = 0.7

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
                    h, w, _ = frame.shape

                    # preprocess img acquired
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
                    img = cv2.resize(img, (640, 480))  # resize
                    img_mean = np.array([127, 127, 127])
                    img = (img - img_mean) / 128
                    img = np.transpose(img, [2, 0, 1])
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32)

                    confidences, boxes = ort_session.run(None, {input_name: img})
                    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                    faces = []
                    for i in range(boxes.shape[0]):
                        box = boxes[i]
                        x1, y1, x2, y2 = box

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        aligned_face = fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                        aligned_face = cv2.resize(aligned_face, (112, 112))
                        print(aligned_face)

                        aligned_face = aligned_face - 127.5
                        aligned_face = aligned_face * 0.0078125

                        faces.append(aligned_face)
                    if len(faces) > 0:
                        predictions = []

                        faces = np.array(faces)
                        feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
                        embeds = sess.run(embeddings, feed_dict=feed_dict)

                        for embedding in embeds:
                            diff = np.subtract(saved_embeds, embedding)
                            dist = np.sum(np.square(diff), 1)
                            idx = np.argmin(dist)
                            if dist[idx] < threshold:
                                predictions.append(names[idx])
                            else:
                                predictions.append("unknown")

                        # draw
                        for i in range(boxes.shape[0]):
                            box = boxes[i]
                            text = str(predictions[i])

                            if text != "unknown":
                                x1, y1, x2, y2 = box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    main()
