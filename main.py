from flask import Flask, render_template, request
import torch
import time
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from lstm_classifier import LSTMClassifier
from config import model_config as config

def get_sentiment(text):
    # Load model
    model = LSTMClassifier(config)
    checkpoint = torch.load('bi_lstm-best_model.pth')

    # Set model to evaluation mode
    model.eval()

    # Define emotion dictionary
    emotion_dict = {
        0: 'anger',  # anger
        1: 'happy',  # happy
        2: 'sadness',  # sadness
        3: 'fear',  # fear
        4: 'surprise',  # surprise
        5: 'neutral'  # neutral
    }
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

    # Prepare input for the model
    max_length = 3033
    vocab_size = 10000
    encoded_sentence = one_hot(text, vocab_size)
    padded_sentence = pad_sequences([encoded_sentence], maxlen=max_length, dtype="long", truncating="post",
                                    padding="post")
    input_ids = torch.from_numpy(padded_sentence).long().float()

    # Reshape input tensor
    input_ids = input_ids.unsqueeze(0)
    print(input_ids)
    print(input_ids.shape)
    # Predict the emotion from the input
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)  # cast input to float32
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_dict[predicted.tolist()[0]]
        print("Predicted emotion:", emotion)
    end_time = time.time()
    # Print time taken for prediction
    return emotion

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        content = request.form.get("text")
        sentiment = get_sentiment(content)
        return "The sentiment is: " + sentiment
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)