from tkinter.ttk import Progressbar

import cv2
import PIL.Image
import PIL.ImageTk
import tkinter
import timeit
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os.path
from imutils import paths
import pickle
from Training import encode, savemodel
from utillities_detection import initialiseDetector, detect
from utilities_recognition import initialiseRecognizer, recognise
from utilities_tracking import track, tagUI, remove_duplicate, update, remove_unknown
import dlib
# from tkinter.ttk import *


class Fyp:
    def __init__(self, window):
        self.window = window
        self.window.title("Final Year Computer Project")
        self.window.configure(bg="black")

        # variables
        self.label_progress = None
        self.track_window = None
        self.team_label = None
        self.label = None
        self.label_topic = None
        self.label_topic2 = None
        self.checker = False
        self.stop_button = None
        self.train_button = None
        self.progress = None
        self.video_stopper = False
        self.vid = None
        self.result = None
        self.app_window = None
        self.window2 = None
        self.window3 = None
        self.window4 = None
        self.window5 = None
        self.photo = None
        self.my_img = None
        self.image_label = None
        self.team_members = None
        self.track_img = None
        self.track_btn = None
        self.counter_test = None
        self.fps_avr = 0
        self.frame_count = 0
        self.trackers = []

        # login window ##################
        self.login = Toplevel()
        self.login.geometry('400x450')

        self.login_img = ImageTk.PhotoImage(Image.open('UI_Components/identity.png'))
        self.image_label = Label(self.login, image=self.login_img)
        self.image_label.pack(pady=(15,0))

        self.login_label = Label(self.login, text='Username', font=('Helvetica', 10))
        self.login_entry = Entry(self.login)
        self.login_label.pack(pady=(5, 0))
        self.login_entry.pack()

        self.login_label1 = Label(self.login, text='Password', font=('Helvetica', 10))
        self.login_entry1 = Entry(self.login, show='*')
        self.login_label1.pack(pady=(5, 0))
        self.login_entry1.pack()

        self.login_btn = Button(self.login, text="Login", width=10, command=self.login_action)
        self.login_btn.pack(pady=(15, 0))

        self.cancel_btn = Button(self.login, text="Cancel", width=10, command=self.cancel_action)
        self.cancel_btn.pack(pady=(15, 0))

        self.error_msg = Label(self.login, text='')
        self.error_msg.pack(pady=(5, 0))
        # ##################################

        # menu bars
        self.menu_bar = Menu(window)
        self.window.config(menu=self.menu_bar)

        # first menu drop down
        self.app_run = Menu(self.menu_bar)
        self.menu_bar.add_cascade(label='Run', menu=self.app_run)
        self.app_run.add_command(label='Train Model', command=self.train_model)
        self.app_run.add_separator()
        self.app_run.add_command(label='Track', command=self.open_track)
        self.app_run.add_separator()
        self.app_run.add_command(label='Open time stamps', command=self.opentime)

        self.app_run.add_separator()
        self.app_run.add_command(label='Open camera', command=self.open_camera)

        # second menu drop down
        self.sub_menu = Menu(self.menu_bar)
        self.menu_bar.add_cascade(label='About', menu=self.sub_menu)
        self.sub_menu.add_command(label='App', command=self.open_app)
        self.sub_menu.add_separator()
        self.sub_menu.add_command(label='Team', command=self.open_team)

        # third menu drop drown
        self.sub_menu2 = Menu(self.menu_bar)
        self.menu_bar.add_cascade(label='Help', menu=self.sub_menu2)
        self.sub_menu2.add_command(label='How to...', command=self.run_app)

        # Label for file dialog
        self.window_label = Label(self.app_window, text='Select File in Mp4 Format', bg='black', fg='white',
                                  font="Courier 12")
        self.window_label.pack(pady=(10, 5))

        # button to open select file dialog
        self.btn = Button(window, text="Select Video", height=2, width=10, command=self.open_file)
        self.btn.pack()

        # canvas for the processing frames
        self.canvas_height = 500
        self.canvas_width = 1000
        self.canvas = tkinter.Canvas(window, height=self.canvas_height, width=self.canvas_width, bg='black')
        self.canvas.pack(pady=(10, 0), padx=(10, 10))
        self.image = PhotoImage(file='UI_Components/play_button.png')
        self.canvas.create_image(320, 90, image=self.image, anchor=NW)

        # button to end the processing frames
        self.btn1 = Button(window, text="End", height=2, width=9, command=self.kill_file)
        self.btn1.pack(pady=(10, 0))

        # pause and play button
        self.btn2 = Button(window, text='Pause', height=2, width=9, command=self.pause_play)
        self.btn2.pack(pady=(10, 10))

        self.delay = 1
        self.fps = 0.0
        self.ort_session, self.input_name = initialiseDetector()
        self.track = []
        self.button = []
        try:
            self.recognizer, self.le, (self.saved_embeds, self.names) = initialiseRecognizer()
        except FileNotFoundError:
            self.recognizer = None
            self.le = None
            self.saved_embeds, self.names = None, None

        self.redetect = -1
        self.face_locations = []
        self.face_names = []
        self.probability = []
        self.pre_frame = None
        self.false_track = {}
        self.redetect_freqeunt = 15
        self.frame_no = 0
        self.present = []
        self.f = None
        self.second = 0
        self.minute = 0
        self.second2 = 0
        self.timestamp = "00:00"

        self.update()
        self.window.withdraw()
        self.window.mainloop()

    # login functionality
    def login_action(self):
        if self.login_entry.get() == '' and self.login_entry1.get() == '':
            self.window.deiconify()
            self.login.destroy()
        else:
            self.error_msg.config(text='Wrong Password/Username')
            self.login_entry.delete(0, 'end')
            self.login_entry1.delete(0, 'end')

    # cancel functionality
    def cancel_action(self):
        self.login.destroy()
        self.window.destroy()
        sys.exit()

    # window for App
    def open_app(self):
        self.window2 = Toplevel()
        self.window2.title('About the App')
        self.window2.iconbitmap(r"UI_Components/favicon.ico")
        self.window2.geometry('400x400')
        self.my_img = ImageTk.PhotoImage(Image.open('UI_Components/identity.png'))
        self.image_label = Label(self.window2, image=self.my_img)
        self.image_label.pack(pady=(50, 0))
        self.label_topic2 = Label(self.window2, text='Video Tagging Based On Person' + '\n' + 'Identification',
                                  font="Courier 15")
        self.label_topic2.pack(pady=(20, 0))

    # window for Team
    def open_team(self):
        self.window3 = Toplevel()
        self.window3.title('About the Team')
        self.window3.iconbitmap(r"UI_Components/favicon.ico")
        self.window3.geometry('200x200')
        self.team_label = Label(self.window3, text='Team 5A',
                                font="Courier 15 bold")
        self.team_label.pack(pady=(20, 0))
        self.team_members = Label(self.window3, text='Ting Teck Chuan' + '\n' + 'Ishan Jeetun' + '\n' + 'Tan Kai Yi',
                                  font="Courier 15")
        self.team_members.pack(pady=(20, 0))

    # window for How to run App
    def run_app(self):
        self.window4 = Toplevel()
        self.window4.title('How to run the App')
        self.window4.iconbitmap(r"UI_Components/favicon.ico")

        file = open("inputs/help.txt")
        data = file.read()
        file.close()
        results = Label(self.window4, text=data)
        results.pack()

    # function to open time stamps
    def opentime(self):
        self.window5 = Toplevel()
        self.window5.title('Time stamps')
        self.window5.iconbitmap(r"UI_Components/favicon.ico")

        file = open("outputs/timestamp.txt")
        data = file.read()
        file.close()
        results = Label(self.window5, text=data)
        results.pack()

    # window for model training
    def train_model(self):
        self.app_window = Toplevel()
        self.app_window.title('Train model')
        self.app_window.iconbitmap(r"UI_Components/favicon.ico")
        self.app_window.geometry('400x400')

        self.label_topic = Label(self.app_window, text='Click Train' + '\n' + 'To Start Training the Model',
                                 font="Courier 15 bold")
        self.label_topic.pack(pady=(100, 10))

        self.train_button = Button(self.app_window, text='Train', height=2, width=10, command=self.progress_bar)
        self.train_button.pack(pady=(10, 10))

        self.progress = Progressbar(self.app_window, orient=HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(pady=(0, 10))

        self.label_progress = Label(self.app_window, text="", font=("Courier", 10))
        self.label_progress.pack(pady=(0, 10))

        self.label = Label(self.app_window, text="", font=("Courier", 10))
        self.label.pack(pady=(0, 10))

    # This is to keep track of who to tag in the video at a particular moment
    def tracker(self, name, x):
        if name not in self.track:
            self.track.append(name)
            self.button[x].config(bg='red')
        else:
            self.track.remove(name)
            self.button[x].config(bg='black')

    # window for tracking
    def open_track(self):
        self.track_window = Toplevel()
        self.track_window.title('Tracking...')
        self.track_window.iconbitmap(r"UI_Components/favicon.ico")
        self.button = []
        for i in range(len(self.le.classes_)):
            if self.le.classes_[i] != "unknown":
                self.button.append(Button(self.track_window, text=self.le.classes_[i],
                                                  height=2,
                                                  width=13,
                                                  bg="red", fg="white",
                                                  command= lambda text = self.le.classes_[i], x=i: self.tracker(text, x)))

                self.button[i].pack()
        self.label_topic = Label(self.track_window, text='')
        self.label_topic.pack()

    def do_smth(self):
        self.label_topic.config(text='Yes')

    def progress_bar(self):
        self.progress['value'] = 5
        imagePaths = list(paths.list_images("dataset"))
        names = []
        images = []
        for (i, imagePath) in enumerate(imagePaths):
            message = "processing image " + str(i + 1) + "/" + str(len(imagePaths))
            self.label_progress.config(text=message)
            self.label.config(text=' Training in Progress ' + '\n' + 'Do not exit')
            self.progress['value'] += 5
            self.app_window.update_idletasks()
            name, encoding = encode(i, imagePath, imagePaths)
            if len(encoding) > 0:
                images.append(encoding)
                names.append(name)
        savemodel(names, images)
        self.recognizer = pickle.loads(open("models/recognizer.pkl", "rb").read())
        self.le = pickle.loads(open("models/le.pkl", "rb").read())
        self.label.config(text=' Training Completed ')

    # function that allow users to select and open file within given constraints
    def open_file(self):
        if os.path.isfile('models/le.pkl') and os.path.isfile('models/recognizer.pkl') and os.path.isfile('models/embeddings.pkl'):
            self.window_label.config(text='Select File in Mp4 Format')
            self.result = filedialog.askopenfile(initialdir="/Users/User/Desktop/Testing", title="Select File",
                                                 filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            self.open_track()
            self.f = open("outputs/timestamp.txt", "w")
            for i in range(len(self.le.classes_)):
                self.track.append(self.le.classes_[i])  # adding all persons to track
        else:
            self.window_label.config(text='Certain Files are missing')
            raise AssertionError('File does not exist')

        if not self.result.name.endswith('.mp4'):
            self.window_label.config(text='File selected has to be in Mp4 Format')
            raise AssertionError('File has to be in Mp4 Format')

        if self.result is not None:
            self.window_label.config(text='Select File in Mp4 Format')
            self.vid = VideoCapture(self.result.name)

    def open_camera(self):
        self.vid = VideoCapture(0)
        self.open_track()
        self.f = open("outputs/timestamp.txt", "w")
        for i in range(len(self.le.classes_)):
            self.track.append(self.le.classes_[i])  # adding all persons to track

    # clear the canvas and stop the video
    def kill_file(self):
        self.canvas.delete('all')
        self.track_window.destroy()
        self.button = []
        self.second = 0
        self.minute = 0
        self.second2 = 0
        self.timestamp = "00:00"
        self.redetect = -1
        self.face_locations = []
        self.face_names = []
        self.probability = []
        self.pre_frame = None
        self.false_track = {}
        self.rerecognized_threshold = 0
        self.redetect_freqeunt = 15
        self.frame_no = 0
        self.present = []

        self.f.close()
        self.canvas.create_image(320, 90, image=self.image, anchor=NW)
        if self.vid is not None:
            self.vid.end_video()
        self.vid = None
        self.video_stopper = False
        self.opentime()
        print(self.fps_avr / self.frame_count)

    # pause and play functionality
    def pause_play(self):
        if self.vid is not None:
            if self.video_stopper is False:
                self.video_stopper = True
                self.btn2.config(text='Play')
            else:
                self.video_stopper = False
                self.btn2.config(text='Pause')

    # update the frames in the canvas with a given delay
    def update(self):
        if self.video_stopper is False:
            if self.vid is not None:
                ## frequency of running recognition process
                self.redetect = (self.redetect + 1) % self.redetect_freqeunt
                ret, frame = self.vid.get_frame()

                if self.pre_frame is None:
                    self.pre_frame = frame
                start = timeit.default_timer()

                if frame is not None:

                    ## if it is time to run recognition process, or there is no one recognized yet, run recognition process
                    if self.redetect == 0:
                        ## detect faces from current frame
                        rgb_frame, temp = detect(frame, self.ort_session, self.input_name)
                        cur_names = []
                        cur_prob = []
                        temp, cur_names, cur_prob = recognise(temp, rgb_frame, self.recognizer, self.le, self.names, self.saved_embeds)
                        ## if there is multiple person recognized as the same identity, choose the highest similarity and remove others
                        cur_names, cur_prob, temp = remove_duplicate(cur_names, temp, cur_prob)
                        ## if someone is recognized as unknown in current frame, but was successfully recognized in previous frame, and his/her false track is still less than specific number, update his/her name and locations to current frame
                        cur_names, cur_prob, temp, false_track = update(cur_names, self.face_names, temp, self.face_locations,
                                                                        cur_prob, self.probability, self.false_track)
                        self.face_locations = temp
                        self.face_names = cur_names
                        self.probability = cur_prob
                        self.face_locations, self.face_names, self.probability = remove_unknown(self.face_locations, self.face_names, self.probability)
                        del (self.trackers)
                        self.trackers = []
                        for i in range(len(self.face_locations)):
                            top, right, bottom, left = self.face_locations[i]
                            box = (left, top, right, bottom)
                            # construct a dlib rectangle object from the bounding box
                            # coordinates and then start the correlation tracker

                            t = dlib.correlation_tracker()
                            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                            t.start_track(rgb_frame, rect)
                            self.trackers.append(t)
                    else:
                        ## track the person by using previous frame and current frame face locations
                        results = []
                        for tracker in self.trackers:
                            output = track(tracker, frame)
                            results.append(output)
                        self.face_locations = results

                    frame = tagUI(frame, self.face_locations, self.face_names, self.probability, self.track)

                    # Time stamp here
                    self.frame_no += 1
                    if self.frame_no % 30 == 0:
                        self.second += 1
                        self.minute = self.second // 60
                        self.second2 = self.second % 60
                        if self.minute < 10:
                            minute_str = "0" + str(self.minute)
                        else:
                            minute_str = str(self.minute)

                        if self.second2 < 10:
                            second_str = "0" + str(self.second2)
                        else:
                            second_str = str(self.second2)
                        self.timestamp = minute_str + ":" + second_str

                        if len(self.present) > 0:
                            s = ""
                            for b in self.present:
                                s = s + b + ", "
                            s = self.timestamp + "  " + s[:-2]
                            self.f.write(s)
                            self.f.write("\n")
                            self.present = []
                    else:
                        for a in self.face_names:
                            if a not in self.present and a != "unknown":
                                self.present.append(a)

                    timeUsed = (timeit.default_timer() - start)
                    if timeUsed != 0:
                        self.fps = (1. / timeUsed)
                    else:
                        self.fps = self.fps_avr/self.frame_count
                    self.fps = min(30, self.fps)
                    self.fps_avr += self.fps
                    self.frame_count += 1
                    cur_fps_avr = self.fps_avr/self.frame_count
                    cv2.putText(frame, "FPS: {:.2f}".format(cur_fps_avr), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 0, 0), 2)

                    cv2.putText(frame, self.timestamp, (int(self.vid.w) - 65, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 0, 0), 2)

                    x = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.vid.write(x)
                    self.pre_frame = frame
                    display = cv2.resize(frame, (1000, 500))
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(display))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                else:
                    self.kill_file()

        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, source):
        self.vid = cv2.VideoCapture(source)
        self.video_checker = True
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', source)

        self.w = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.out = cv2.VideoWriter('outputs/output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (int(self.w), int(self.h)))

    def get_frame(self):
        ret, frame = None, None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, frame

    def write(self, frame):
        self.out.write(frame)

    def end_video(self):
        if self.vid.isOpened():
            self.vid.release()


Fyp(Tk())
