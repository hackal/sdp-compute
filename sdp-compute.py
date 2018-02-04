import os
import io
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
import sys
import json

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

    #print("Setting up API...", file=sys.stderr)

    #Authentication to API
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth.json"

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.join(
    os.path.dirname(__file__),
    'classify/' + filename)

    #print("file name: " + file_name, file=sys.stderr)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return processGoogleVisionAPI(labels)

def processGoogleVisionAPI(labels):

    potentialMaterials = set()

    #print('Labels:', file=sys.stderr)
    for label in labels:

        #print(label.description + " " + str(label.score), file=sys.stderr)

        if ("glass" in label.description) and (label.score > 0.7):
            potentialMaterials.add("glass")

        if ("plastic" in label.description) and (label.score > 0.7):
            potentialMaterials.add("plastic")

        if (("aluminum" in label.description) or ("can" in label.description)) and (label.score > 0.7):
            potentialMaterials.add("aluminum")

    #Get rid of "set()" being printed when potentialMaterials is empty and return first
    return list(potentialMaterials)[0]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():

    material = ""

    if request.method == 'POST':

        #print("Posting...", file=sys.stderr)

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            material = setupGoogleVisionAPI(filename)
            #print("material: " + material, file=sys.stderr)
            result = material
            return json.dumps(result)
    #print("Finishing...", file=sys.stderr)
    #resp = flask.Response("Foo bar baz")
    #resp.headers['Access-Control-Allow-Origin'] = '*'
    #return resp
    return ""

if __name__ == "__main__":
    app.run(host='165.227.234.4', port=5000, debug=True)