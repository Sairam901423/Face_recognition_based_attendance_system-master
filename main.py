import tkinter as tk
from tkinter import ttk, messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from pathlib import Path

# Constants and directory setup
BASE_DIR = Path(__file__).parent
DIRS = {
    'data': BASE_DIR / "StudentDetails",
    'training': BASE_DIR / "TrainingImage",
    'model': BASE_DIR / "TrainingImageLabel",
    'attendance': BASE_DIR / "Attendance"
}

for dir_path in DIRS.values():
    dir_path.mkdir(exist_ok=True)

class AttendanceSystem:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition Attendance System")
        self.window.geometry("1280x720")
        self.window.configure(background='#262523')
        self.setup_gui()

    def setup_gui(self):
        # Frame setup
        self.frame1 = tk.Frame(self.window, bg="#00aeff")
        self.frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

        self.frame2 = tk.Frame(self.window, bg="#00aeff")
        self.frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

        # Title
        tk.Label(self.window, 
                text="Face Recognition Based Attendance System",
                fg="white", bg="#262523", width=55, height=1,
                font=('times', 29, 'bold')).place(x=10, y=10)

        # Clock frame
        clock_frame = tk.Frame(self.window, bg="#c4c6ce")
        clock_frame.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

        self.clock_label = tk.Label(clock_frame, fg="orange", bg="#262523",
                                  width=55, height=1, font=('times', 22, 'bold'))
        self.clock_label.pack(fill='both', expand=1)
        self.update_clock()

        # Setup main content
        self.setup_registration_frame()
        self.setup_attendance_frame()
        self.setup_menubar()

    def setup_registration_frame(self):
        # Registration section header
        tk.Label(self.frame2, 
                text="For New Registrations",
                fg="black", bg="#3ece48",
                font=('times', 17, 'bold')).grid(row=0, column=0)

        # Student ID input
        tk.Label(self.frame2, text="Enter ID", width=20, height=1,
                fg="black", bg="#00aeff",
                font=('times', 17, 'bold')).place(x=80, y=55)

        self.id_entry = tk.Entry(self.frame2, width=32, fg="black",
                               font=('times', 15, 'bold'))
        self.id_entry.place(x=30, y=88)

        # Name input
        tk.Label(self.frame2, text="Enter Name", width=20,
                fg="black", bg="#00aeff",
                font=('times', 17, 'bold')).place(x=80, y=140)

        self.name_entry = tk.Entry(self.frame2, width=32, fg="black",
                                 font=('times', 15, 'bold'))
        self.name_entry.place(x=30, y=173)

        # Buttons
        tk.Button(self.frame2, text="Take Images", command=self.take_images,
                 fg="white", bg="blue", width=34, height=1,
                 activebackground="white",
                 font=('times', 15, 'bold')).place(x=30, y=300)

        tk.Button(self.frame2, text="Save Profile", command=self.save_profile,
                 fg="white", bg="blue", width=34, height=1,
                 activebackground="white",
                 font=('times', 15, 'bold')).place(x=30, y=380)

        self.message_label = tk.Label(self.frame2, text="", bg="#00aeff",
                                    fg="black", width=39, height=1,
                                    font=('times', 16, 'bold'))
        self.message_label.place(x=7, y=450)

    def setup_attendance_frame(self):
        # Attendance section header
        tk.Label(self.frame1, text="For Already Registered",
                fg="black", bg="#3ece48",
                font=('times', 17, 'bold')).place(x=0, y=0)

        # Treeview for attendance
        self.attendance_tree = ttk.Treeview(self.frame1, height=13,
                                          columns=('name', 'date', 'time'))
        self.attendance_tree.column('#0', width=82)
        self.attendance_tree.column('name', width=130)
        self.attendance_tree.column('date', width=133)
        self.attendance_tree.column('time', width=133)
        self.attendance_tree.grid(row=2, column=0, padx=(0,0),
                                pady=(150,0), columnspan=4)

        # Scrollbar
        scroll = ttk.Scrollbar(self.frame1, orient='vertical',
                             command=self.attendance_tree.yview)
        scroll.grid(row=2, column=4, padx=(0,100),
                   pady=(150,0), sticky='ns')
        self.attendance_tree.configure(yscrollcommand=scroll.set)

        # Buttons
        tk.Button(self.frame1, text="Take Attendance",
                 command=self.track_images, fg="black", bg="yellow",
                 width=35, height=1, activebackground="white",
                 font=('times', 15, 'bold')).place(x=30, y=50)

        tk.Button(self.frame1, text="Quit",
                 command=self.window.destroy, fg="black", bg="red",
                 width=35, height=1, activebackground="white",
                 font=('times', 15, 'bold')).place(x=30, y=450)

    def setup_menubar(self):
        menubar = tk.Menu(self.window, relief='ridge')
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Change Password',
                           command=self.change_password)
        filemenu.add_command(label='Contact Us',
                           command=lambda: mess._show(title='Contact',
                           message='Contact at: support@example.com'))
        filemenu.add_command(label='Exit',
                           command=self.window.destroy)
        menubar.add_cascade(label='Help', menu=filemenu)
        self.window.config(menu=menubar)

    def update_clock(self):
        time_string = time.strftime('%H:%M:%S')
        self.clock_label.config(text=time_string)
        self.window.after(1000, self.update_clock)

    def take_images(self):
        if not self.validate_inputs():
            return

        try:
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            student_id = self.id_entry.get()
            name = self.name_entry.get()
            serial = self.get_next_serial()

            cap = cv2.VideoCapture(0)
            sample_num = 0

            while sample_num < 100:
                ret, img = cap.read()
                if not ret:
                    raise Exception("Camera access failed")

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num += 1
                    
                    # Save face image
                    img_path = DIRS['training'] / f"{name}.{serial}.{student_id}.{sample_num}.jpg"
                    cv2.imwrite(str(img_path), gray[y:y + h, x:x + w])

                cv2.imshow('Taking Images', img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            # Save student details
            self.save_student_details(serial, student_id, name)
            self.message_label.config(text="Images Captured Successfully!")

        except Exception as e:
            mess._show(title='Error',
                      message=f'Error capturing images: {str(e)}')

    def save_profile(self):
        if not Path('haarcascade_frontalface_default.xml').exists():
            mess._show(title='Error',
                      message='Haarcascade file is missing!')
            return

        try:
            faces, ids = [], []
            for img_path in DIRS['training'].glob('*.jpg'):
                pil_img = Image.open(img_path).convert('L')
                img_np = np.array(pil_img, 'uint8')
                id_num = int(img_path.stem.split('.')[1])
                faces.append(img_np)
                ids.append(id_num)

            if not faces:
                raise Exception("No training images found")

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.save(str(DIRS['model'] / 'Trainner.yml'))
            
            self.message_label.config(
                text=f"Profile Saved Successfully! ({len(set(ids))} faces)"
            )

        except Exception as e:
            mess._show(title='Error',
                      message=f'Error saving profile: {str(e)}')

    def track_images(self):
        if not self.check_prerequisites():
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(DIRS['model'] / 'Trainner.yml'))
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            df = pd.read_csv(DIRS['data'] / 'StudentDetails.csv')
            cap = cv2.VideoCapture(0)
            attendance_marked = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
                    id_num, conf = recognizer.predict(gray[y:y + h, x:x + w])

                    if conf < 50 and id_num not in attendance_marked:
                        attendance_marked.add(id_num)
                        self.mark_attendance(id_num, df)

                cv2.imshow('Taking Attendance', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            mess._show(title='Error',
                      message=f'Error tracking attendance: {str(e)}')

    def mark_attendance(self, id_num, df):
        try:
            name = df.loc[df['SERIAL NO.'] == id_num]['NAME'].values[0]
            date = datetime.datetime.now().strftime('%d-%m-%Y')
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')

            attendance_file = DIRS['attendance'] / f"Attendance_{date}.csv"
            attendance = [str(id_num), '', name, '', date, '', timestamp]

            mode = 'a+' if attendance_file.exists() else 'w'
            with open(attendance_file, mode, newline='') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(['Id', '', 'Name', '', 'Date', '', 'Time'])
                writer.writerow(attendance)

            # Update treeview
            self.attendance_tree.insert('', 0, text=str(id_num),
                                     values=(name, date, timestamp))

        except Exception as e:
            mess._show(title='Error',
                      message=f'Error marking attendance: {str(e)}')

    def validate_inputs(self):
        student_id = self.id_entry.get()
        name = self.name_entry.get()

        if not student_id or not name:
            mess._show(title='Error',
                      message='Please fill all fields!')
            return False

        if not name.replace(' ', '').isalpha():
            mess._show(title='Error',
                      message='Name should only contain alphabets!')
            return False

        return True

    def get_next_serial(self):
        try:
            student_file = DIRS['data'] / 'StudentDetails.csv'
            if not student_file.exists():
                pd.DataFrame(columns=['SERIAL NO.', '', 'ID', '', 'NAME']).to_csv(
                    student_file, index=False)
                return 1
                
            df = pd.read_csv(student_file)
            return (len(df) // 2) + 1
        except Exception:
            return 1

    def save_student_details(self, serial, student_id, name):
        student_file = DIRS['data'] / 'StudentDetails.csv'
        new_row = pd.DataFrame([[serial, '', student_id, '', name]],
                             columns=['SERIAL NO.', '', 'ID', '', 'NAME'])
        new_row.to_csv(student_file, mode='a', header=False, index=False)

    def check_prerequisites(self):
        if not Path('haarcascade_frontalface_default.xml').exists():
            mess._show(title='Error',
                      message='Haarcascade file is missing!')
            return False

        if not (DIRS['model'] / 'Trainner.yml').exists():
            mess._show(title='Error',
                      message='Model not trained! Please register students first.')
            return False

        if not (DIRS['data'] / 'StudentDetails.csv').exists():
            mess._show(title='Error',
                      message='No student details found!')
            return False

        return True

    def change_password(self):
        self.master = tk.Toplevel(self.window)
        self.master.geometry("400x160")
        self.master.resizable(False, False)
        self.master.title("Change Password")
        self.master.configure(background="white")
        
        # Old password entry
        tk.Label(self.master, text='Enter Old Password',
                bg='white', font=('times', 12, 'bold')).place(x=10, y=10)
        self.old_pass = tk.Entry(self.master, width=25, fg="black",
                                relief='solid', font=('times', 12, 'bold'),
                                show='*')
        self.old_pass.place(x=180, y=10)
        
        # New password entry
        tk.Label(self.master, text='Enter New Password',
                bg='white', font=('times', 12, 'bold')).place(x=10, y=45)
        self.new_pass = tk.Entry(self.master, width=25, fg="black",
                                relief='solid', font=('times', 12, 'bold'),
                                show='*')
        self.new_pass.place(x=180, y=45)
        
        # Confirm new password
        tk.Label(self.master, text='Confirm New Password',
                bg='white', font=('times', 12, 'bold')).place(x=10, y=80)
        self.confirm_pass = tk.Entry(self.master, width=25, fg="black",
                                   relief='solid', font=('times', 12, 'bold'),
                                   show='*')
        self.confirm_pass.place(x=180, y=80)
        
        # Buttons
        tk.Button(self.master, text="Cancel",
                 command=self.master.destroy, fg="black", bg="red",
                 height=1, width=25, activebackground="white",
                 font=('times', 10, 'bold')).place(x=200, y=120)
                 
        tk.Button(self.master, text="Save",
                 command=self.save_password, fg="black", bg="#3ece48",
                 height=1, width=25, activebackground="white",
                 font=('times', 10, 'bold')).place(x=10, y=120)
    
    def save_password(self):
        # Read existing password
        try:
            with open(DIRS['model'] / "psd.txt", "r") as f:
                stored_pass = f.read().strip()
        except FileNotFoundError:
            stored_pass = None
            
        old = self.old_pass.get()
        new = self.new_pass.get()
        confirm = self.confirm_pass.get()
        
        if stored_pass and old != stored_pass:
            mess._show(title='Wrong Password',
                      message='Please enter correct old password.')
            return
            
        if new != confirm:
            mess._show(title='Error',
                      message='Confirm new password again!!!')
            return
            
        try:
            with open(DIRS['model'] / "psd.txt", "w") as f:
                f.write(new)
            mess._show(title='Success',
                      message='Password changed successfully!!')
            self.master.destroy()
        except Exception as e:
            mess._show(title='Error',
                      message=f'Error saving password: {str(e)}')

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    app = AttendanceSystem()
    app.run()
