import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage


from nada_dsl import *


from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from nada_dsl import *
import warnings
import tensorflow as tf
import logging

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# Load models once
vgg_model = load_model("/mnt/f/nadaquickstart/nadaproject/resrc/vgg_model.h5")
vgg_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

resnet_model = load_model("/mnt/f/nadaquickstart/nadaproject/resrc/resnet_model.h5")
resnet_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (cap.get(cv2.CAP_PROP_FRAME_COUNT) // num_frames) == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            frames.append(frame)
        frame_count += 1
    cap.release()
    frames = np.concatenate(frames, axis=0)
    return frames



@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join('/tmp', video.filename)
    video.save(video_path)

    print(f"Video saved at: {video_path}")  # Debugging statement

    # Preprocess the input video frames
    input_frames = extract_frames(video_path, num_frames=10)
    input_frames = np.array(input_frames)
    input_frames = input_frames / 255.0
    vgg_predictions = vgg_model.predict(input_frames, verbose=0)
    resnet_predictions = resnet_model.predict(input_frames, verbose=0)

    print(f"VGG Predictions: {vgg_predictions}")  # Debugging statement
    print(f"ResNet Predictions: {resnet_predictions}")  # Debugging statement

    # Count the number of predictions for each class (0: real, 1: fake)
    class_counts = np.bincount([np.argmax(vgg_predictions), np.argmax(resnet_predictions)])

    # Get the index of the class with the highest count
    prediction = int(np.argmax(class_counts))

    print(f"Final Prediction: {prediction}")  # Debugging statement

    
         
    output_data={"result":prediction}


    current_directory = os.getcwd()
    output_file = os.path.join(current_directory, "output.json")
    
    
    import json

    with open(output_file,"w") as file:
        json.dump(output_data,file) 
    
    print("written successfully!")

    return jsonify({'prediction': "real" if prediction == 0 else 'fake'})



if __name__ == '__main__':
    app.run(debug=True)