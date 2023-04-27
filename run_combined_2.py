import torch
import time
import numpy as np
import librosa
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from librosa.feature import mfcc
from lstm_classifier import LSTMClassifier
from config import model_config as config

#Load model
model = LSTMClassifier(config)
checkpoint = torch.load('bi_lstm-best_model.pth')

#Fix size mismatch error
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

#Set model to evaluation mode
model.eval()

#Define emotion dictionary
emotion_dict = {
    0: 'ang', # anger
    1: 'hap', # happy
    2: 'sad', # sadness
    3: 'fea', # fear
    4: 'sur', # surprise
    5: 'neu' # neutral
}
#Prepare input for the model
text_input = "and what are you going to do about it ? you going to call someone ? you going to send me somewhere else ?"
audio_input = 'Ses03M_impro05a_M012.wav'
signal, sr = librosa.load(audio_input, sr=16000)
#Cảm xúc phải nhận dạng tức giận

#Prepare text input
max_length_text = 3033
vocab_size = 10000
encoded_text = one_hot(text_input, vocab_size)
padded_text = pad_sequences([encoded_text], maxlen=max_length_text, dtype="long", truncating="post", padding="post")
text_input_ids = torch.from_numpy(padded_text).long().float()

#Reshape text input tensor
text_input_ids = text_input_ids.unsqueeze(0)

#Split audio into frames and extract MFCC features
mfccs = []
frame_length = int(0.025 * sr)
frame_stride = int(0.01 * sr)
for i in range(0, len(signal), frame_stride):
    if i + frame_length < len(signal):
        frame = signal[i:i + frame_length]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        mfccs.append(mfcc.flatten())

#Pad MFCC features to fixed length
max_length_audio = 3033
padded_mfccs = np.zeros((max_length_audio, len(mfccs[0])))
if len(mfccs) > max_length_audio:
    mfccs = mfccs[:max_length_audio]
for i in range(len(mfccs)):
    padded_mfccs[i, :] = mfccs[i]
audio_input_mfccs = torch.tensor(padded_mfccs)


#Add an additional dimension to the input tensor
audio_input_mfccs = audio_input_mfccs.unsqueeze(0)

#Update size of input tensor
batch_size = 1
input_size = audio_input_mfccs.shape[2]
max_length = audio_input_mfccs.shape[1]
expanded_size = (batch_size, max_length, input_size)
audio_input_mfccs = audio_input_mfccs.expand(expanded_size)

#Add an additional dimension before transposing
audio_input_mfccs = audio_input_mfccs.unsqueeze(3)

#Transpose the tensor
audio_input_mfccs = audio_input_mfccs.transpose(1, 2)

#Remove the additional dimension
audio_input_mfccs = audio_input_mfccs.squeeze(3)
# Combine text and audio features
combined_input = torch.cat((text_input_ids, audio_input_mfccs.float()), dim=1)

# Predict emotion
start_time = time.time()
with torch.no_grad():
    # Cast input to float32
    combined_input = combined_input.float()
    outputs = model(combined_input)
    _, predicted = torch.max(outputs, 1)
    emotion = emotion_dict[predicted.tolist()[0]]
    print("Predicted emotion:", emotion)
end_time = time.time()

# Print time taken for prediction
print(f"Time taken for prediction: {end_time - start_time:.2f} seconds")