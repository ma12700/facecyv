import cv2
import face_recognition
import numpy
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/run/',methods=['POST'])
def run():
    keysDic = request.files.keys()
    if len(keysDic) != 2:
        return jsonify({"message": "must send two images"})

    keys = []
    for k in keysDic:
        keys.append(k)

    filestr = request.files[keys[0]].read()
    npimg = numpy.fromstring(filestr, numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings1 = face_recognition.face_encodings(rgb, boxes)

    filestr = request.files[keys[1]].read()
    npimg = numpy.fromstring(filestr, numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings2 = face_recognition.face_encodings(rgb, boxes)
    if(len(encodings1) != 1 or len(encodings2) != 1):
        return jsonify({"message": "no faces"})
    matches = face_recognition.compare_faces(encodings1, encodings2[0], tolerance=0.5)
    if(matches[0]):
        return jsonify({"message": "matched"})
    else:
        return jsonify({"message": "not matched"})

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)