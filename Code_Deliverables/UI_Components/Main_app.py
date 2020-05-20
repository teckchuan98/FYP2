from tkinter.ttk import Progressbar

import cv2
import PIL.Image
import PIL.ImageTk
import tkinter
import time
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os.path

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
        self.photo = None
        self.my_img = None
        self.image_label = None
        self.team_members = None
        self.track_img = None
        self.track_btn = None
        self.counter_test = None

        # login window ##################
        self.login = Toplevel()
        self.login.geometry('400x450')

        self.login_img = ImageTk.PhotoImage(Image.open('identity.png'))
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
        self.image = PhotoImage(file='play_button.png')
        self.canvas.create_image(320, 90, image=self.image, anchor=NW)

        # button to end the processing frames
        self.btn1 = Button(window, text="End", height=2, width=9, command=self.kill_file)
        self.btn1.pack(pady=(10, 0))

        # pause and play button
        self.btn2 = Button(window, text='Pause', height=2, width=9, command=self.pause_play)
        self.btn2.pack(pady=(10, 10))

        self.delay = 1
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
        self.window2.iconbitmap(r"favicon.ico")
        self.window2.geometry('400x400')
        self.my_img = ImageTk.PhotoImage(Image.open('identity.png'))
        self.image_label = Label(self.window2, image=self.my_img)
        self.image_label.pack(pady=(50, 0))
        self.label_topic2 = Label(self.window2, text='Video Tagging Based On Person' + '\n' + 'Identification',
                                  font="Courier 15")
        self.label_topic2.pack(pady=(20, 0))

    # window for Team
    def open_team(self):
        self.window3 = Toplevel()
        self.window3.title('About the Team')
        self.window3.iconbitmap(r"favicon.ico")
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
        self.window4.iconbitmap(r"favicon.ico")

    # window for model training
    def train_model(self):
        self.app_window = Toplevel()
        self.app_window.title('How to run the App')
        self.app_window.iconbitmap(r"favicon.ico")
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

    # window for tracking##############
    def open_track(self):
        self.track_window = Toplevel()
        self.track_window.title('Tracking...')
        self.track_window.iconbitmap(r"favicon.ico")

        for i in range(4):

            # self.track_img = PhotoImage(file=r'identity.png')
            # self.track_img = self.track_img.subsample(3, 3)

            self.track_btn = Button(self.track_window, text='Bla bla', command=self.do_smth)
            self.track_btn.pack()

        self.label_topic = Label(self.track_window, text='')
        self.label_topic.pack()

    def do_smth(self):
        self.label_topic.config(text='Yes')
    # ###########################################

    def progress_bar(self):
        self.progress['value'] = 5
        self.counter_test = 0
        while self.progress['value'] < 100:
            self.counter_test += 1
            self.label_progress.config(text=self.counter_test)
            self.label.config(text=' Training in Progress ' + '\n' + 'Do not exit')
            self.progress['value'] += 5
            self.app_window.update_idletasks()
            time.sleep(1.0)
        self.label.config(text=' Training Completed ')

    # function that allow users to select and open file within given constraints
    def open_file(self):
        if os.path.isfile('fyp.txt'):
            self.window_label.config(text='Select File in Mp4 Format')
            self.result = filedialog.askopenfile(initialdir="/Users/User/Desktop/Testing", title="Select File",
                                                 filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        else:
            self.window_label.config(text='Certain Files are missing')
            raise AssertionError('File does not exist')

        if not self.result.name.endswith('.mp4'):
            self.window_label.config(text='File selected has to be in Mp4 Format')
            raise AssertionError('File has to be in Mp4 Format')

        if self.result is not None:
            self.window_label.config(text='Select File in Mp4 Format')
            self.vid = VideoCapture(self.result.name)

    # clear the canvas and stop the video
    def kill_file(self):
        self.canvas.delete('all')
        self.canvas.create_image(320, 180, image=self.image, anchor=CENTER)
        if self.vid is not None:
            self.vid.end_video()
        self.vid = None
        self.video_stopper = False

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
                ret, frame = self.vid.get_frame()
                if frame is not None:
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
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
        #self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                  # (int(self.w), int(self.h)))

    def get_frame(self):
        ret, frame = None, None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            #self.out.write(frame)
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
