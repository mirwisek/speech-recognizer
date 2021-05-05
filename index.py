import librosa
import soundfile
import pickle
import numpy as np
# Just need traceback for debugging
import traceback
from os import path
from flask import Flask, request, json, Response

app = Flask(__name__)

# full path of current script
file_path = path.abspath(__file__)
# full path of the directory of current script
dir_path = path.dirname(file_path)
MODEL_PATH = path.join(dir_path, 'KNN_classifier.model')

#Emotions to observe
observed_emotions=['happy', 'sad', 'neutral', 'angry']

model = pickle.load(open(MODEL_PATH, 'rb'))

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        print(f'The sample rate is {sample_rate}')
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result


def preprocess(file):
  return extract_feature(file, mfcc=True, chroma=True, mel=True)


def send_result(response=None, error='', status=200):
    if response is None: response = {}
    result = json.dumps({'result': response, 'error': error})
    return Response(status=status, mimetype="application/json", response=result)


@app.route('/')
def hello():
    return "Welcome to Privacy Gaurd, are you lost?"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['audio']
        res = preprocess(file)
        result = model.predict([res])[0]
        return send_result(result)
    except KeyError:
        return send_result(error='An audio file is required', status=422)
    except Exception as e:
        traceback.print_exc()
        return send_result(error='Prediction Error', status=500)
  
    
if __name__ == '__main__':
    app.run()
    