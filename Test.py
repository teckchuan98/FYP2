import cv2
import PIL.Image
import PIL.ImageTk
import tkinter
import time
from tkinter import *
from tkinter import filedialog


class Fyp:
    def __init__(self, window):
        self.window = window
        self.window.title("Final Year Computer Project")
        self.window.iconbitmap(r"favicon.ico")
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

        self.canvas = tkinter.Canvas(window, height=360, width=640)
        self.canvas.pack(pady=(10, 0), padx=(10, 500))
        self.image = PhotoImage(file='play_button.png')
        self.canvas.create_image(320, 180, image=self.image, anchor=CENTER)
        self.btn1 = Button(window, text="End", height=2, width=13, bg="black", fg="white", command=self.kill_file)
        self.btn1.pack(pady=(5, 5), padx=(10, 500))
        self.btn1 = Button(window, text="Pause", height=2, width=13, bg="black", fg="white", command=self.pause_play)
        self.btn1.pack(pady=(5, 5), padx=(10, 500))
        self.btn3 = Button(window, text="Test", height=2, width=13, bg="black", fg="white", command=self.test)
        self.btn3.pack(pady=(5, 5), padx=(10, 500))
        self.delay = 15
        self.update()
        self.window.mainloop()

    def test(self):
        if self.vid is None:
            print('Train Now')

    def kill_file(self):
        self.canvas.delete('all')
        self.canvas.create_image(320, 180, image=self.image, anchor=CENTER)
        if self.vid is not None:
            self.vid.end_video()
        self.vid = None
        self.video_stopper = False

    def open_file(self):

        #if recognizer present:

        self.result = filedialog.askopenfile(initialdir="/Users/User/Desktop/Testing", title="Select File",
                                             filetypes=(("png files", "*.png"), ("mp4 files", "*.mp4"), ("text files",
                                                                                                         "*.txt"), (
                                                 "all files", "*.*")))
        self.vid = VideoCapture(self.result.name)
        #else:
                #display click on train button

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
                    #processing
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
        self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(self.w), int(self.h)))

    def get_frame(self):
        ret, frame = None, None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            # timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")
            #             # print(timestamp)
            self.out.write(frame)
            if ret:
                return ret,  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, frame

    def end_video(self):
        if self.vid.isOpened():
            self.vid.release()


Fyp(Tk())