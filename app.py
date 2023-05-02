import numpy as np
import pickle
import base64
from flask import Flask, request, render_template, jsonify
from imutils.paths import list_images
import mediapipe as mp
import os
import io
import cv2
from PIL import Image
from model.SiameseModel import SiameseModel
from tensorflow import keras

# configured paths (change if needed)
IMAGES_PATH = "Register/original"
CROPPED_IMAGES_PATH = "Register/cropped"
MODEL_PATH = "model/"

# Create application
app = Flask(__name__)

def save_image_from_base64(base64_str, save_path):
    """
    Save base64 string to image file.
    Args:  
        base64_str (str): base64 string.
        save_path (str): path to save image file.
    Returns:
        str: path to saved image file.
    """
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # save the image to disk
    img.save(save_path, 'JPEG')

    return save_path


def register_to_model(firstName, lastName, imagePath):
    """
    Register a person to the database.
    Args:
        firstName (str): first name of the person.
        lastName (str): last name of the person.
        imagePath (str): path to the person's photo.
    Returns:
        None
    """

    print("[INFO] Registering to the model...")
    # Load the model
    model = SiameseModel(network=keras.models.load_model(filepath=MODEL_PATH))

    # Extract data from arguments
    name = firstName + "_" + lastName
    image = Image.open(imagePath)

    # Register to the model
    model.register(np.array(image), name)

    print("[INFO] Done...")


# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# Bind predict function to URL
@app.route('/register', methods=['POST'])

def register():
    """
    Register a person to the database.
    Args:
        first_name (str): first name of the person.
        last_name (str): last name of the person.
        photo_base64 (str): base64 string of the person's photo.
    Returns:
        dict: contains the message, first name, last name, and photo filename.
    """
    # get the first name, last name, and photo from the request
    first_name = request.form.get('firstName')
    last_name = request.form.get('lastName')
    photo_base64 = request.form.get('imagePreview')

    if(first_name is None or last_name is None or photo_base64 is None):
        print("[ERROR] Some/all of the parameters is None")
        return jsonify({
            'message': 'Photo processing unsuccessful',
            'first_name': first_name,
            'last_name': last_name,
            'photo_filename': photo_base64
        })

    # debug
    # print(first_name)
    # print(last_name)
    # print(photo_base64)

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

            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
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
        print("[INFO] Done (img saved at " + facePath + ")... ")
    
    else:
        print("[INFO] Directory does not exist... cannot crop images")

    # Register the person to the model
    #register_to_model(first_name, last_name, facePath)

    # return a JSON response with the results
    return jsonify({
        'message': 'Photo processing successful',
        'first_name': first_name,
        'last_name': last_name,
        'photo_filename': photo
    })


if __name__ == '__main__':
    app.run(debug=True)
