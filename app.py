import numpy as np
import pickle
import tqdm
from flask import Flask, request, render_template, jsonify
from imutils.paths import list_images
import mediapipe as mp
import os
import cv2

# Create application
app = Flask(__name__)

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
    photo = request.files.get('imagePreview')

    print(first_name)
    print(last_name)
    print(photo)

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

    if photo is not None:
        # Save original image from the form to this path
        photo.save(os.path.join(dirPath, name + "_0001.jpg"))

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
        print("[ERROR] No photo was uploaded")

    # return a JSON response with the results
    return jsonify({
        'message': 'Photo processing successful',
        'first_name': first_name,
        'last_name': last_name,
        'photo_filename': photo.filename
    })

if __name__ == '__main__':
    app.run(debug=True)
