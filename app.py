import numpy as np
import pickle
import base64
from flask import Flask, request, render_template, jsonify
from imutils.paths import list_images
import mediapipe as mp
import os
import io
import cv2
from io import BytesIO
from PIL import Image

# Create application
app = Flask(__name__)

def save_image_from_base64(base64_str, save_path):
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # save the image to disk
    img.save(save_path, 'JPEG')

    return save_path


# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# Bind predict function to URL
@app.route('/register', methods=['POST'])

def register():
    # get the first name, last name, and photo from the request
    first_name = request.form.get('firstName')
    last_name = request.form.get('lastName')
    photo_base64 = request.form.get('imagePreview')

    # print(first_name)
    # print(last_name)
    # print(photo)

    IMAGES_PATH = "Register/original"
    CROPPED_IMAGES_PATH = "Register/cropped"

    # Make sure the directories exist
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    if not os.path.exists(CROPPED_IMAGES_PATH):
        os.makedirs(CROPPED_IMAGES_PATH)

    # do some processing on the photo (save it to disk)
    mp_face_detection = mp.solutions.face_detection

    print("[INFO] Cropping the images and saving them...")

    # Create a folder with the person's name and thje original image in it
    name = first_name + "_" + last_name
    dirPath = os.path.join(IMAGES_PATH, name)
    print("Working on " + dirPath)
    print(dirPath)
    
    if not os.path.exists(dirPath):
        print("[INFO] Path does not exist, creating...")
        os.makedirs(dirPath)

    # Save original image from the form to this path
    #file = capture_photo(first_name=first_name, last_name=last_name, filepath=IMAGES_PATH)
    file = save_image_from_base64(photo_base64, os.path.join(dirPath, first_name + "_" + last_name + ".jpg"))
    if(file is None):
        print("[ERROR] Unable to capture from webcam... ")
        return jsonify({
            'message': 'Photo processing unsuccessful',
            'first_name': first_name,
            'last_name': last_name,
            'photo_filename': file
        })
    
    # Load file from 'file' variable
    photo = file

    if os.path.isdir(dirPath):
        imagePaths = list(list_images(dirPath))
        outputDir = os.path.join(CROPPED_IMAGES_PATH, name)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # loop over the image paths
        for img_path in imagePaths:
            imageID = img_path.split(os.path.sep)[-1]
            with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5) as face_detection:

                image = cv2.imread(img_path)
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                for i, detection in enumerate(results.detections):
                    x1, y1 = int(detection.location_data.relative_bounding_box.xmin * image.shape[1]), int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                    x2, y2 = int((detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width) * image.shape[1]), int((detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height) * image.shape[0])
                    cropped_image = image[y1:y2, x1:x2]
                    facePath = os.path.sep.join([outputDir, imageID])

                    # Check for empty images
                    if cropped_image.size == 0:
                        continue
                    cv2.imwrite(facePath, cropped_image)
        print("[INFO] Done... ")
    
    else:
        print("[INFO] Directory does not exist... cannot crop images")


    # return a JSON response with the results
    return jsonify({
        'message': 'Photo processing successful',
        'first_name': first_name,
        'last_name': last_name,
        'photo_filename': photo
    })

if __name__ == '__main__':
    app.run(debug=True)
