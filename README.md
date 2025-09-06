# Face Recognition Attendance System

This project is a real-time attendance system using facial recognition, featuring liveness detection via blinking to prevent spoofing attacks.

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>


### 2. Create the environment using Python 3.11
py -3.11 -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### 3.
pip install -r requirements.txt


### 4. Download Required Files

The system requires two pre-trained model files that must be downloaded manually and placed in the main project folder.

Dlib Landmark Predictor: This is required for blink detection.
Download: [shape_predictor_68_face_landmarks.dat.bz2](https://www.google.com/search?q=http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
Unzip the file and place shape_predictor_68_face_landmarks.dat in the project root.


VGG-Face Model Weights: This is for face recognition.
Download: [vgg_face_weights.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5)
Place vgg_face_weights.h5 inside a folder path: C:\Users\<Your-Username>\.deepface\weights\.

 Create these folders if they don't exist.


###5. Enroll Students

 # Example for a student with ID 101
python enroll.py --id 101 --folder "path/to/student_101_images/"


###6. Run the app
python run_attendance.py


