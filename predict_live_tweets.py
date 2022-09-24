
import numpy as np
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', return_dict=False)
MAX_LEN = 80  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_tweets(model, df, risk_level=0.2):
  
    trade_bool = False
    polarity = 0
    min_certainty = 7 - (risk_level*4)
    predictions = []
    for tweet in df.tweet:
        encoded_tweet = tokenizer.encode_plus(tweet, max_length=MAX_LEN, 
        add_special_tokens=True, return_token_type_ids=False, padding=True, 
        return_attention_mask=True, return_tensors='pt')
        input_ids = encoded_tweet['input_ids'].to(device)
        attention_mask = encoded_tweet['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        max_score, prediction = torch.max(output, dim=1)
        predictions.append(prediction.item())
        min_score, _ = torch.min(output, dim=1)
        certainty = (max_score - min_score).item()
        if certainty >= min_certainty:
            if prediction.item() == 0:
                polarity -= 1
            else:
                polarity += 1
    if polarity > 0 or polarity < 0:
        trade_bool = True

    df['prediction'] = np.array(predictions)

    return polarity, trade_bool, df
