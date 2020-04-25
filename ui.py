import cv2
import PIL.Image
import PIL.ImageTk
import tkinter
from tkinter import *
from tkinter import filedialog

import numpy as np
import pickle
import cv2
import face_recognition
import onnxruntime as ort
from detector import detect



class Fyp:
    def __init__(self, window):
        self.window = window
        self.window.title("Final Year Computer Project")
        # self.window.iconbitmap(r"favicon.ico")
        self.window.configure(bg="black")
        self.video_stopper = False
        self.vid = None
        self.result = None
        self.photo = None

        self.my_label = Label(window, text="Team 5A", bg="black", fg="white").pack(pady=(10, 0), padx=(10, 500))
        self.my_label1 = Label(window, text="Video Tagging Based on Person Identification", bg="black", fg="white"). \
            pack(pady=(0, 10), padx=(10, 500))

        self.btn = Button(window, text="Select Video", command=self.open_file)
        self.btn.pack(padx=(10, 500))

        self.canvas = tkinter.Canvas(window, height=550, width=1080)
        self.canvas.pack(pady=(10, 0), padx=(10, 500))

        self.image = PhotoImage(file='play1.png')

        self.canvas.create_image(0, 0, image=self.image, anchor="nw")

        self.btn1 = Button(window, text="End", height=2, width=13, bg="black", fg="white", command=self.kill_file)
        self.btn1.pack(side=LEFT)

        self.btn2 = Button(window, text="Pause", height=2, width=13, bg="black", fg="white", command=self.pause_play)
        self.btn2.pack(side=LEFT)

        self.delay = 1

        self.ort_session = ort.InferenceSession('ultra_light_640.onnx')  # load face detection model
        self.input_name = self.ort_session.get_inputs()[0].name

        self.recognizer = pickle.loads(open("recognizer.pkl", "rb").read())
        self.le = pickle.loads(open("le.pkl", "rb").read())

        self.video_capture = cv2.VideoCapture('chandler.mp4')
        with open("embeddings.pkl", "rb") as f:
            (saved_embeds, names) = pickle.load(f)

        w = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(w), int(h)))

        self.track = []
        self.button = []
        for i in range(len(self.le.classes_)):
            if self.le.classes_[i] != "unknown":
                self.button.append(Button(window, text=self.le.classes_[i], height=2, width=13, bg="black", fg="white",  command= lambda text = self.le.classes_[i], x=i: self.tracker(text, x)))
                self.button[i].pack(side=RIGHT)

        print(self.track)


        self.update()
        self.window.mainloop()

    def tracker(self, name, x):
        if name not in self.track:
            self.track.append(name)
            self.button[x].config(bg='red')
        else:
            self.track.remove(name)
            self.button[x].config(bg='black')

    def test(self):
        if self.vid is None:
            print('Train Now')

    def kill_file(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")
        if self.vid is not None:
            self.vid.end_video()
        self.vid = None
        self.video_stopper = False

    def open_file(self):

        # if recognizer present:

        self.result = filedialog.askopenfile(initialdir="/Users/User/Desktop/Testing", title="Select File",
                                             filetypes=(("png files", "*.png"), ("mp4 files", "*.mp4"), ("text files",
                                                                                                         "*.txt"), (
                                                            "all files", "*.*")))
        self.vid = VideoCapture(self.result.name)
        # else:
        # display click on train button

    def pause_play(self):
        if self.vid is not None:
            if self.video_stopper is False:
                self.video_stopper = True
            else:
                self.video_stopper = False

    def update(self):
        if self.video_stopper is False:
            if self.vid is not None:
                ret, frame = self.vid.get_frame()
                if frame is not None:
                    # processing
                    boxes, labels, probs = detect(frame, self.ort_session, self.input_name)
                    face_locations = []
                    for i in boxes:
                        x1, y1, x2, y2 = i
                        y = (y1, x2, y2, x1)
                        face_locations.append(y)
                    rgb_frame = frame[:, :, ::-1]

                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    face_names = []
                    probability = []

                    for face_encoding in face_encodings:

                        face_encoding = [face_encoding]
                        preds = self.recognizer.predict_proba(face_encoding)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = self.le.classes_[j]
                        if proba > 0.5:
                            face_names.append(name)
                            probability.append(proba)
                        else:
                            face_names.append("unknown")
                    print(self.track)
                    for (top, right, bottom, left), name, prob in zip(face_locations, face_names, probability):
                        if name == "unknown":
                            continue

                        x = prob * 100
                        x = str(x)
                        x = x[:3]
                        x = x + "%"

                        # Draw a box around the face
                        if name in self.track:
                            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                            cv2.putText(frame, name + " : " + x, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

                    frame = cv2.resize(frame, (1080, 550))

                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                else:
                    self.kill_file()

        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, source):
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', source)

        self.w = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (int(self.w), int(self.h)))

    def get_frame(self):
        ret, frame = None, None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            # timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")
            #             # print(timestamp)
            self.out.write(frame)
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, frame

    def end_video(self):
        if self.vid.isOpened():
            self.vid.release()


Fyp(Tk())
