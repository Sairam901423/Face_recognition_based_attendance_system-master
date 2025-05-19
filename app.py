import streamlit as st
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from pathlib import Path

# Get the face recognizer using different possible methods
def get_face_recognizer():
    try:
        # Try the standard method first
        return cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            # Try alternative import
            return cv2.face_LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                # Try older OpenCV method
                return cv2.createLBPHFaceRecognizer()
            except AttributeError:
                st.error("Could not initialize face recognition. Please check OpenCV installation.")
                return None

# Set up page config and constants
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👥",
    layout="wide",
)

# Constants
DATA_DIR = Path("StudentDetails")
TRAINING_DIR = Path("TrainingImage")
LABEL_DIR = Path("TrainingImageLabel")
ATTENDANCE_DIR = Path("Attendance")

# Create necessary directories
for dir_path in [DATA_DIR, TRAINING_DIR, LABEL_DIR, ATTENDANCE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    /* CSS Variables for theming */
    :root {
        --background-color: #f8f9fa;
        --text-color: #1e3d59;
        --card-background: white;
        --border-color: #e0e5e9;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #1e1e1e;
            --text-color: #ffffff;
            --card-background: #2d2d2d;
            --border-color: #404040;
        }
    }

    /* Global Styles */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Title Styles with green highlight */
    .title-text {
        text-align: center;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #4CAF50 0%, #81c784 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #4CAF50;
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Button Styles */
    .stButton > button {
        background-color: #4CAF50;
        color: white !important;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Header Styles */
    .stats-container {
        padding: 1.5rem;
        background: var(--card-background);
        color: var(--text-color);
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background-color: var(--card-background);
    }

    .css-1d391kg .stSelectbox label {
        color: var(--text-color) !important;
    }
    
    /* Metric Cards */
    .st-emotion-cache-1xarl3l {
        background: var(--card-background);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50;
    }
    
    /* Success Messages */
    .st-emotion-cache-1rm05kb {
        background-color: #4CAF50;
        color: white !important;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Error Messages */
    .st-emotion-cache-16idsys {
        background-color: #ff6b6b;
        color: white !important;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Info Messages */
    .st-emotion-cache-1vbkxwb {
        background-color: #4361ee;
        color: white !important;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    /* DataFrames */
    .st-emotion-cache-1ylmt1q {
        background: var(--card-background);
        color: var(--text-color);
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        overflow: hidden;
    }
    
    /* Forms */
    .stTextInput > div > div {
        background-color: var(--card-background);
        color: var(--text-color);
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        padding: 0.5rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Date Input */
    .stDateInput > div {
        background-color: var(--card-background);
        color: var(--text-color);
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }

    /* Additional text color fixes */
    .stMarkdown, .stText {
        color: var(--text-color) !important;
    }

    /* Table text color fix */
    .dataframe {
        color: var(--text-color) !important;
    }

    /* Input text color fix */
    input, textarea, select {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

def get_images_and_labels(path):
    """Get training images and their corresponding labels."""
    image_paths = list(Path(path).glob("*.jpg"))
    faces = []
    ids = []
    
    for image_path in image_paths:
        # Convert image to grayscale
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        
        # Extract ID from filename
        id_num = int(image_path.stem.split(".")[1])
        
        faces.append(image_np)
        ids.append(id_num)
        
    return faces, ids

def train_model():
    """Train the face recognition model."""
    try:
        faces, ids = get_images_and_labels(TRAINING_DIR)
        if not faces:
            return False, "No training images found"
            
        recognizer = get_face_recognizer()
        if recognizer is None:
            return False, "Could not initialize face recognition"
            
        recognizer.train(faces, np.array(ids))
        recognizer.save(str(LABEL_DIR / "Trainner.yml"))
        return True, f"Model trained successfully with {len(faces)} images"
    except Exception as e:
        print(f"DEBUG: Error type: {type(e)}, Error message: {str(e)}")
        return False, f"Error training model: {str(e)}"

def get_student_details():
    """Get student details from CSV file with proper structure"""
    if not (DATA_DIR / "StudentDetails.csv").exists():
        # Create new CSV with proper structure
        df = pd.DataFrame(columns=['SERIAL_NO', 'ID', 'NAME'])
        df.to_csv(DATA_DIR / "StudentDetails.csv", index=False)
        return df
    
    df = pd.read_csv(DATA_DIR / "StudentDetails.csv")
    # Clean up any empty columns if they exist
    if len(df.columns) > 3:
        df = df.iloc[:, [0, 2, 4]]  # Select only the needed columns
        df.columns = ['SERIAL_NO', 'ID', 'NAME']
        df.to_csv(DATA_DIR / "StudentDetails.csv", index=False)
    return df

def main():
    st.markdown('<h1 class="title-text">Face Recognition Based Attendance System</h1>', unsafe_allow_html=True)
    
    menu = ["Home", "Take Attendance", "Register New Student", "View Records"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("System Statistics")
            df = get_student_details()
            total_students = len(df)
            st.metric("Total Students Registered", total_students)
            
        with col2:
            st.subheader("Today's Attendance")
            today = datetime.datetime.now().strftime('%d-%m-%Y')
            attendance_file = ATTENDANCE_DIR / f"Attendance_{today}.csv"
            if attendance_file.exists():
                df = pd.read_csv(attendance_file)
                st.metric("Students Present Today", len(df) // 2)
            else:
                st.metric("Students Present Today", 0)
                
        with col3:
            st.subheader("Model Status")
            model_file = LABEL_DIR / "Trainner.yml"
            if model_file.exists():
                st.success("Face Recognition Model: Ready")
            else:
                st.warning("Face Recognition Model: Not Trained")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif choice == "Take Attendance":
        st.subheader("Take Attendance")
        
        # Check if model exists
        if not (LABEL_DIR / "Trainner.yml").exists():
            st.error("Face recognition model not found. Please register and train with some students first.")
            return
            
        if st.button("Start Camera"):
            if not Path("haarcascade_frontalface_default.xml").exists():
                st.error("Haarcascade file missing!")
                return
            
            cap = cv2.VideoCapture(0)
            recognizer = get_face_recognizer()
            recognizer.read(str(LABEL_DIR / "Trainner.yml"))
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            
            # Create placeholders
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            stop_button = st.button("Stop Camera")
            
            attendance_recorded = set()  # Track recorded attendance
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    id_num, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    
                    # Increased confidence threshold from 50 to 70 for better recognition
                    if conf < 70:
                        df = get_student_details()
                        try:
                            name = df[df['SERIAL_NO'] == id_num]['NAME'].values[0]
                            
                            # Mark attendance if not already recorded
                            if id_num not in attendance_recorded:
                                ts = time.time()
                                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                                
                                attendance_file = ATTENDANCE_DIR / f"Attendance_{date}.csv"
                                attendance = [str(id_num), name, date, timestamp]  # Removed empty columns
                                
                                # Create or append to attendance file
                                mode = 'a+' if attendance_file.exists() else 'w'
                                with open(attendance_file, mode, newline='') as f:
                                    writer = csv.writer(f)
                                    if mode == 'w':
                                        writer.writerow(['Id', 'Name', 'Date', 'Time'])  # Simplified headers
                                    writer.writerow(attendance)
                                
                                attendance_recorded.add(id_num)
                                status_placeholder.success(f"✅ Attendance marked for {name}")
                        except IndexError:
                            cv2.putText(frame, "Unknown ID", (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            continue
                    else:
                        cv2.putText(frame, "Unknown", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")
            
            cap.release()
            
    elif choice == "Register New Student":
        st.subheader("Register New Student")
        
        with st.form("registration_form"):
            student_id = st.text_input("Enter Student ID")
            student_name = st.text_input("Enter Student Name")
            submitted = st.form_submit_button("Register")
            
            if submitted and student_id and student_name:
                if not Path("haarcascade_frontalface_default.xml").exists():
                    st.error("Haarcascade file missing!")
                    return
                    
                if not student_name.replace(" ", "").isalpha():
                    st.error("Please enter a valid name (only alphabets and spaces allowed)")
                    return
                    
                # Get next serial number
                df = get_student_details()
                serial = len(df) + 1
                
                # Start face capture
                cap = cv2.VideoCapture(0)
                detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                
                st.info("Camera starting... Look at the camera and wait...")
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                sample_num = 0
                
                while sample_num < 100:
                    ret, img = cap.read()
                    if not ret:
                        break
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        sample_num += 1
                        
                        # Save captured face
                        face_img = gray[y:y + h, x:x + w]
                        img_path = TRAINING_DIR / f"{student_name}.{serial}.{student_id}.{sample_num}.jpg"
                        cv2.imwrite(str(img_path), face_img)
                        
                        # Update UI
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(img_rgb)
                        progress_bar.progress(sample_num/100)
                
                cap.release()
                
                # Save student details
                new_row = pd.DataFrame([[serial, student_id, student_name]], 
                                     columns=['SERIAL_NO', 'ID', 'NAME'])
                new_row.to_csv(DATA_DIR / "StudentDetails.csv", mode='a', header=False, index=False)
                
                # Train model
                success, message = train_model()
                if success:
                    st.success(f"Registration Successful! {message}")
                else:
                    st.error(f"Registration completed but model training failed: {message}")
                    
    elif choice == "View Records":
        st.subheader("Attendance Records")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_date = st.date_input(
                "Select Date",
                datetime.datetime.now()
            ).strftime('%d-%m-%Y')
        
        attendance_file = ATTENDANCE_DIR / f"Attendance_{selected_date}.csv"
        if attendance_file.exists():
            try:
                df = pd.read_csv(attendance_file)
                
                # No need to clean columns since CSV already has correct structure
                df_clean = df.copy()
                
                # Sort by time to show latest entries first
                df_clean['Time'] = pd.to_datetime(df_clean['Time'], format='%H:%M:%S')
                df_clean = df_clean.sort_values('Time', ascending=False)
                df_clean['Time'] = df_clean['Time'].dt.strftime('%H:%M:%S')
                
                with col2:
                    st.download_button(
                        "📥 Download Attendance",
                        df_clean.to_csv(index=False),
                        f"attendance_{selected_date}.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                # Display total attendance count
                st.info(f"Total attendance for {selected_date}: {len(df_clean)} students")
                
                # Display the dataframe with improved styling
                st.dataframe(
                    df_clean,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Id": st.column_config.NumberColumn("ID", help="Student ID"),
                        "Name": st.column_config.TextColumn("Name", help="Student Name"),
                        "Date": st.column_config.DateColumn("Date", help="Attendance Date"),
                        "Time": st.column_config.TimeColumn("Time", help="Attendance Time")
                    }
                )
            except Exception as e:
                st.error(f"Error reading attendance file: {str(e)}")
        else:
            st.info("No attendance records found for selected date")

if __name__ == '__main__':
    main()