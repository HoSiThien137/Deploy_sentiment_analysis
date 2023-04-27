import torch
import pickle
from lstm_classifier import LSTMClassifier
from config import model_config as config
from utils import load_data


# Load test data
test_pairs = load_data(test=True)
inputs, targets = test_pairs
inputs = inputs.unsqueeze(0)

# Load pretrained model
model = LSTMClassifier(config)
checkpoint = torch.load('E:\app_demo_text\bi_lstm-best_model.pth'.format(config['model_code']),
                        map_location='cpu')
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    # Predict
    predict_probas = model(inputs).cpu().numpy()

    with open('"E:\app_emtion\combined_lstm_classifier.pkl"', 'wb') as f:
        pickle.dump(predict_probas, f)