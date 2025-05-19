# Face Recognition Based Attendance System

A modern face recognition-based attendance system built with Python and Streamlit, providing an intuitive web interface for managing student attendance.

## Features

- 🎯 Modern, responsive web interface built with Streamlit
- 🔒 Secure and efficient face recognition using OpenCV
- 📊 Real-time attendance tracking and visualization
- 💾 Daily attendance records in CSV format
- 👥 Easy student registration with face capture
- 📈 View and download attendance reports
- 🌓 Supports both light and dark modes
- ⚡ Live camera feed for attendance marking
- 📁 Git LFS support for large model files

## Technology Stack

- **Frontend:** Streamlit for the web interface
- **Computer Vision:** OpenCV and face recognition (cv2.face.LBPHFaceRecognizer)
- **Data Processing:** Pandas, NumPy
- **Storage:** CSV for student details and attendance records
- **Version Control:** Git with LFS support for large files

## Directory Structure

```
├── app.py                  # Main application file
├── haarcascade_frontalface_default.xml  # Face detection model
├── StudentDetails/         # Student registration data
├── TrainingImage/         # Student face images
├── TrainingImageLabel/    # Trained model files
└── Attendance/           # Daily attendance records
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Face_recognition_based_attendance_system.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### 1. Home Page
- View system statistics
- Check today's attendance count
- Monitor model status

### 2. Register New Student
- Enter student details
- Capture face images
- Automatic model training

### 3. Take Attendance
- Real-time face recognition
- Automatic attendance marking
- Live status updates

### 4. View Records
- Browse attendance by date
- Download attendance reports
- View student details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note

The system uses Git LFS for managing large model files. Make sure to install Git LFS before cloning the repository.
