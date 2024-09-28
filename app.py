import fire
import cv2
import numpy as np
import json
import tensorflow as tf
import urllib.request
import os

new_model = tf.keras.models.load_model('3dresnet_model.keras')

def download_video(video_url):
    temp_video_path = "temp_video.mp4"
    try:
        urllib.request.urlretrieve(video_url, temp_video_path)
        return temp_video_path
    except Exception as e:
        print("Error downloading video:", e)
        return None

def predict_video_labels(video_url, n_frames=10, height=224, width=224):
    video_path = download_video(video_url)
    if video_path is None:
        return None  

    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frames.append(frame / 255.0)  
    cap.release()

    while len(frames) < n_frames:
        frames.append(np.zeros((height, width, 3)))

    frames = np.array(frames)
    frames = frames.reshape((1, n_frames, height, width, 3)) 

    predictions = new_model.predict(frames)
    predicted_probabilities = predictions[0]

    labels = {0: 'basketball', 1: 'baseball', 2: 'cricket', 3: 'beach soccer', 4: 'golf'}
    probabilities_per_label = {labels[i]: float(predicted_probabilities[i]) for i in range(len(predicted_probabilities))}

    max_index = np.argmax(predicted_probabilities)
    max_probability = predicted_probabilities[max_index]
    max_label = labels[max_index]

    result_json = json.dumps({
        'predicted_probabilities': probabilities_per_label,
        'max_label': max_label,
        'max_probability': float(max_probability)
    })

    os.remove(video_path) 
    return result_json

if __name__ == '__main__':
    fire.Fire(predict_video_labels)
