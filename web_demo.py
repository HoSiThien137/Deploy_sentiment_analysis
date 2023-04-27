from flask import Flask, render_template, request
import torch
import numpy as np
import librosa
from librosa.feature import mfcc
from lstm_classifier import LSTMClassifier
from config import model_config as config
from sklearn.feature_extraction.text import TfidfVectorizer


def get_sentiment(text, audio):
    # Load model
    model = LSTMClassifier(config)
    checkpoint = torch.load('thu1.pth')

    # Fix size mismatch error
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name.startswith('rnn.weight_ih_l0'):
            # Replace weight_ih_l0
            new_weight_ih_l0 = torch.zeros_like(model_state_dict[name])
            new_weight_ih_l0[:1024, :3033] = param[:, :3033]
            model_state_dict[name] = new_weight_ih_l0
        elif name.startswith('rnn.weight_ih_l0_reverse'):
            # Replace weight_ih_l0_reverse
            new_weight_ih_l0_reverse = torch.zeros_like(model_state_dict[name])
            new_weight_ih_l0_reverse[:1024, :3033] = param[:, :3033]
            model_state_dict[name] = new_weight_ih_l0_reverse

    model.load_state_dict(model_state_dict)

    # Set model to evaluation mode
    model.eval()

    # Define emotion dictionary
    emotion_dict = {
        0: 'Anger😠🤬',  # anger
        1: 'Happy😁😊',  # happy
        2: 'Sadness😔😢',  # sadness
        3: 'Fear😨😱',  # fear
        4: 'Surprise😮😮',  # surprise
        5: 'Neutral😐🤨'  # neutral
    }
    # Prepare input for the model
    sr = 44100
    y, sr = librosa.load(audio, sr=sr)

    # Prepare text input`
    max_length_text = 3025
    tfidf_vectorizer = TfidfVectorizer(max_features=max_length_text)
    encoded_text = tfidf_vectorizer.fit_transform([text])
    text_feature = torch.from_numpy(encoded_text.toarray()).float()
    padding_length = max_length_text - text_feature.shape[1]
    text_feature = np.pad(text_feature, ((0, 0), (0, padding_length)), mode='constant')
    text_feature = text_feature.reshape((1, -1))

    # Calculate signal mean
    rmse = librosa.feature.rms(y=y)[0]
    # list feature
    sig_mean = np.mean(np.abs(y))
    sig_std = np.std(y)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)
    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    y_harmonic = librosa.effects.hpss(y)[0]
    harmonic = np.mean(y_harmonic) * 1000
    cl = 0.45 * sig_mean
    center_clipped = []
    for s in y:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    auto_corr_max = 1000 * np.max(auto_corrs) / len(auto_corrs)
    auto_corr_std = np.std(auto_corrs)

    # Combine these features into a single feature vector
    audio_feature = [sig_mean, sig_std, rmse_mean, rmse_std, silence, harmonic, auto_corr_max, auto_corr_std]

    # Combine text and audio features
    audio_feature = np.array(audio_feature, dtype=np.float32)
    text_feature = np.array(text_feature, dtype=np.float32)
    combined_input = np.concatenate((text_feature.reshape(1, -1), np.reshape(audio_feature, (1, -1))), axis=1)
    combined_input = np.expand_dims(combined_input, axis=0)  # Thêm một chiều
    print(text_feature)
    print(audio_feature)
    print(combined_input)
    # Predict emotion
    with torch.no_grad():
        combined_input = combined_input.astype('float32')
        combined_input = torch.from_numpy(combined_input)
        outputs = model(combined_input)
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_dict[predicted.tolist()[0]]
        return emotion


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        content = request.form.get("text")
        audio = request.form.get("voice")
        sentiment = get_sentiment(content, audio)
        return render_template('index.html', sentiment=sentiment)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
