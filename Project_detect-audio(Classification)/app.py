import numpy as np
from flask import Flask,render_template, request
from preprocess import prediction

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    fileaudio = request.files["audio_file"]
    audio_path = "./static/"+ fileaudio.filename
    fileaudio.save(audio_path)

    audio = "."+ audio_path
    pred_label = prediction(file=audio_path)
    return render_template("index.html" , prediction=pred_label, file=audio)

@app.route('/list_labels')
def list_labels():
    list_label = np.load('list_labels.npy')
    return render_template('list_labels.html',list_label=list_label)

if __name__ == "__main__":
    app.run(port=3000, debug=True)