import os
import io
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
import sys
import json

#Imports for rotating image
import cv2
import numpy as np

#Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

UPLOAD_FOLDER = 'classify'
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg', 'gif'])

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(
    PROPAGATE_EXCEPTIONS = True
)

def setupGoogleVisionAPI(filename):

    #Authentication to API
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth.json"

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.join(
    os.path.dirname(__file__),
    'classify/' + filename)

    print("file name: " + file_name)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return processGoogleVisionAPI(labels)

def processGoogleVisionAPI(labels):

    potentialMaterials = {}

    for label in labels:

        print(label.description + ": " + str(label.score))

        if ("glass" in label.description) and (label.score > 0.6):
            if label.score > potentialMaterials.get("glass", 0.00):
                potentialMaterials['glass'] = label.score

        if ("plastic" in label.description) and (label.score > 0.6):
            if label.score > potentialMaterials.get("plastic", 0.00):
                potentialMaterials['plastic'] = label.score

        if (("aluminum" in label.description) or ("can" in label.description)) and (label.score > 0.6):
            if label.score > potentialMaterials.get("aluminum", 0.00):
                potentialMaterials['aluminum'] = label.score

    print("Potential materials: ")
    print(potentialMaterials)

    material = {}
    if len(potentialMaterials) > 0:
        v=list(potentialMaterials.values())
        k=list(potentialMaterials.keys())
        material[k[v.index(max(v))]] = max(v)
        print("Dictionary: ", material)

    return material


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():

    material = ""

    if request.method == 'POST':

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #Rotate image so that the vision API looks at the image from
            #multiple angles.
            os.chdir('classify')
            img = cv2.imread(filename)
            num_rows, num_cols = img.shape[:2]

            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
            rotation_matrix1 = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 270, 1)
            img_rotation1 = cv2.warpAffine(img, rotation_matrix1, (num_cols, num_rows))

            cv2.imwrite('classify1.jpg',img_rotation)
            cv2.imwrite('classify2.jpg',img_rotation1)
            os.chdir('..')

            #Combine the response from the upright photo and the flipped photo
            materials = {}
            materials.update(setupGoogleVisionAPI('classify1.jpg'))
            materials.update(setupGoogleVisionAPI('classify2.jpg'))

            v=list(materials.values())
            k=list(materials.keys())
            result = []
            if len(v) > 0:
                for key, value in materials.items():
                    maxPrediction = max(v)
                    if value > (maxPrediction-0.1):
                        result.append(key)


            print("result: ", result)
            return json.dumps(result)
    return ""

@app.route("/commands", methods=['GET', 'POST'])
def commands():
    if request.method == 'POST':
        command = request.form.get("data")
        print(command)
        f = open('command.txt', 'w' )
        f.write(command)
        f.close()
        return json.dumps("recieved")

    if request.method == 'GET':
        f = open('command.txt', 'r')
        command = f.read()
        #Empty the text file so command is not called again
        f = open('command.txt', 'w' ).close()
        return json.dumps(command)


    return ""

if __name__ == "__main__":
    app.run(host='165.227.234.4', port=5000, debug=True)
