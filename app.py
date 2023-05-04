import numpy as np
import pickle
import base64
from flask import Flask, request, render_template, jsonify, Response
from imutils.paths import list_images
from imutils.video import VideoStream
import mediapipe as mp
import os
import io
import time
import signal
import threading
import cv2
from PIL import Image
from model.SiameseModel import SiameseModel
from tensorflow import keras

# configured paths (change if needed)
MODEL_PATH = "model/saved_model"

# create application
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()
# define video frames
raw_video_frame = None
output_video_frame = None
video_active = True

vs = VideoStream(src=0).start()
time.sleep(2.0)
# define siamese model & mediapipe face detection
siamese_model = SiameseModel(network=keras.models.load_model(filepath=MODEL_PATH))
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# add signal handler to handle server shutdown
# def shutdown_handler(signum, frame):
#     global video_active
#     video_active = False    

# signal.signal(signal.SIGINT, shutdown_handler)

# create frame generators for streaming video to browser
def gen_raw_frame():  # generate raw frame
    global raw_video_frame
    while True:
        if raw_video_frame is None:
            continue
        
        ret, buffer = cv2.imencode('.jpg', raw_video_frame)

        if not ret:
            continue
        
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(buffer) + b'\r\n')


def gen_output_frame():  # generate raw frame
    global output_video_frame
    while True:
        if output_video_frame is None:
            continue
        
        ret, buffer = cv2.imencode('.jpg', output_video_frame)

        if not ret:
            continue
        
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(buffer) + b'\r\n')


def encode_cvimage_base64(cv_image):
    _, im_arr = cv2.imencode('.jpg', cv_image)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64    

def decode_cvimage_base64(base64Str):
    im_bytes = base64.b64decode(base64Str.split(',')[1])
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    cv_image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return cv_image


# bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# expose endpoint for raw video
@app.route('/raw')
def raw():
    return Response(gen_raw_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# expose endpoint for output video
@app.route('/output')
def output():
    return Response(gen_output_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# bind predict function to URL
@app.route('/capture', methods=['POST'])    
def capture():
    global raw_video_frame
    if raw_video_frame is not None:            
        print("[SUCCESS] Photo captured")
        image_base64 = encode_cvimage_base64(raw_video_frame)
        return jsonify({
            'message': 'success: photo captured',
            'imageBase64': image_base64.decode('utf8')
        })
    else:
        print("[ERROR] Could not get frame from webcam")
        return jsonify({
            'message': 'error: could not get frame from webcam',
        })

# bind predict function to URL
@app.route('/register', methods=['POST'])
def register():
    global raw_video_frame, face_detection, siamese_model, lock
    """
    Register a person to the database.
    Args:
        first_name (str): first name of the person.
        last_name (str): last name of the person.
        image_base64 (str): image data in base64
    Returns:
        dict: contains the message, first name, last name, and photo filename.
    """
    # get the first name, last name from the request
    first_name = request.form.get('firstName')
    last_name = request.form.get('lastName')
    image_base64 = request.form.get('imagePreview')

    #print(f"first_name: {first_name}, last_name: {last_name}, image : {image}")

    if (first_name is None or last_name is None or image_base64 is None):
        print("[ERROR] Some/all of the parameters is None")
        return jsonify({
            'message': 'error: incomplete input form',
        })
    else:
        image = decode_cvimage_base64(image_base64)
        name = f"{first_name}_{last_name}"
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # extract faces
        results = face_detection.process(rgb_image)

        if (len(results.detections) > 0):
            bbox = results.detections[0].location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * rgb_image.shape[1]), int(bbox.ymin * rgb_image.shape[0])
            x2, y2 = int((bbox.xmin + bbox.width) * rgb_image.shape[1]), int((bbox.ymin + bbox.height) * rgb_image.shape[0])
            cropped_face = rgb_image[y1:y2, x1:x2]
            with lock:
                siamese_model.register(cropped_face, name)        
            print(f"[INFO] New face registered ({name})")
            return jsonify({
                'message': 'success: new face added to model',
            })
        else:
            print("[ERROR] No face detected")
            return jsonify({
                'message': 'error: no face detected',
            })


def get_frame_from_webcam():
    global vs, raw_video_frame

    while True:        
        frame = vs.read()
        if frame is not None:
            raw_video_frame = frame.copy()


def perform_face_recognition():
    global vs, raw_video_frame, output_video_frame, face_detection, siamese_model, lock

    while True:        
        if raw_video_frame is None:
            continue
        cv_image = raw_video_frame.copy()
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        if (results.detections is None):
            continue
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * rgb_image.shape[1]), int(bbox.ymin * rgb_image.shape[0])
            x2, y2 = int((bbox.xmin + bbox.width) * rgb_image.shape[1]), int((bbox.ymin + bbox.height) * rgb_image.shape[0])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255,0,0), 2)
            cropped_face = rgb_image[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue
            if siamese_model.registered_faces.keys:
                with lock:
                    detected_name, distance = siamese_model.recognize_face(cropped_face, return_distance=True)
                    cv2.putText(cv_image, f"{detected_name} ({distance:.3f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        output_video_frame = cv_image.copy()

    
if __name__ == '__main__':

    # start
    webcam_thread = threading.Thread(target=get_frame_from_webcam)
    webcam_thread.daemon = True
    webcam_thread.start()

    face_recognition_thread = threading.Thread(target=perform_face_recognition)
    face_recognition_thread.daemon = True
    face_recognition_thread.start()

    # start server
    app.run(debug=True, threaded=True, use_reloader=False)

vs.stop()

    
