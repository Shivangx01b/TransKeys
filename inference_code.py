import torch
import torch.nn as nn
from fastapi import FastAPI
from keras.preprocessing.sequence import pad_sequences
import pickle
from pydantic import BaseModel


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_length, output_dim, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encodings = nn.Parameter(torch.zeros(max_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encodings[:x.size(1), :]
        transformer_out = self.transformer_encoder(embedded.permute(1, 0, 2))
        out = self.fc(transformer_out[-1, :, :])
        return out

vocab_size = 40
d_model = 512
nhead = 8
num_encoder_layers = 3
dim_feedforward = 2048
max_length = 40  
output_dim = 3
dropout = 0.1

with open('tokenizer_transformer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    max_length=max_length,
    output_dim=output_dim,
    dropout=dropout
).to(device)


model.load_state_dict(torch.load('model_nn_transformer2.pth', map_location=device))
model.eval()

def predict_question(question, tokenizer, model, max_length, device):
    
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)

    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

class Sentence(BaseModel):
    sentence: str

app = FastAPI()



@app.post("/predict/")
async def predict(sentence_data: Sentence):
    sentence = sentence_data.sentence
    predictions = []
    for word in sentence.split():
        prediction = predict_question(word, tokenizer, model, 40, device)
        if prediction == 0:
            predictions.append((word, 'AWS Access Key'))
        elif prediction == 1:
            predictions.append((word, 'AWS Secret Key'))
    if len(predictions) == 0:
        return {"No AWS keys found ": sentence}
    else:
        return {"sentence": sentence, "prediction": predictions}






