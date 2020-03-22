import numpy as np
import pickle
import cv2
import face_recognition


def main():
    video_capture = cv2.VideoCapture('rdj.mp4')
    with open("embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)

    print(saved_embeds)
    while True:
        ret, frame = video_capture.read()
        if frame is not None:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                diff = np.subtract(saved_embeds, face_encoding)
                dist = np.sum(np.square(diff), axis=1)
                idx = np.argmin(dist)

                if dist[idx] < 0.6:
                    face_names.append(names[idx])
                else:
                    face_names.append("unknown")

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name == "unknown":
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
