import numpy as np
import pickle
import cv2
from Compute_detections import predict
import onnxruntime as ort


def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
    input_name = ort_session.get_inputs()[0].name

    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")  # load face embedding model

    recognizer = pickle.loads(open("recognizer_model.pickle", "rb").read())   # load svm pickle file and label encoder
    le = pickle.loads(open("label_encoder.pickle", "rb").read())

    video_capture = cv2.VideoCapture('test.mp4')
    w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

            for i in range(boxes.shape[0]):
                box = boxes[i]
                x1, y1, x2, y2 = box

                crop = frame[y1:y2, x1:x2]
                h1, w1, _ = crop.shape

                try:
                    blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(blob)
                    vec = embedder.forward()

                    # classificationq
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    probability = preds[j]
                    name = le.classes_[j]

                    text = "{}: {:.2f}%".format(name, probability * 100)

                    if probability > 0.9:
                        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                except Exception as e:
                    print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)



if __name__ == "__main__":
    main()
