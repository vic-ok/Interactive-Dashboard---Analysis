#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import pandas as pd
import subprocess

# Install tf-keras if not already installed
try:
    import tf_keras
except ImportError:
    subprocess.check_call(['pip', 'install', 'tf-keras'])

from transformers import pipeline
from huggingface_hub import login

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

#Roberta Pretrained Model From Hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Login to Hugging Face (you can skip this if you have already logged in)
login(token='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')

#Read dataFrame
df = pd.read_csv('Breakfast Cooking - Kids Game - Copy - Copy.csv', encoding='utf-8')

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(ex):
    encoded_text = tokenizer(ex, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
        
    scores_dict = {'roberta_neg':scores[0],
                  'roberta_neu':scores[1],
                  'roberta_pos':scores[2],
                  'roberta_polarity': 1 if scores[2]>scores[0] else 0
                  }
    return scores_dict

#Run the polarity scores on the dataFrame
res = {}
robertaArr = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Content']
        myId = row['Id']
 

        roberta_result = polarity_scores_roberta(text)
        robertaArr[myId] = roberta_result
        both=  {**roberta_result}
        res[myId] = both
    except RuntimeError:
        print(f'Broke for {myId}')
        

        
roberta = pd.DataFrame(robertaArr).T
roberta = roberta.reset_index().rename(columns={'index': 'Id'})
roberta = roberta.merge(df, how='left')


# Load the emotion analysis pipeline using EmoRoBERTa
emotion_pipeline = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

data = roberta

def get_emotional_label(text):
  return (emotion_pipeline(text)[0]['label'])

data['emotion'] = data['Content'].apply(get_emotional_label)

# Save DataFrame to CSV
csv_file = "Gumin_BreakFastCooking_emotionNew.csv"
data.to_csv(csv_file, index=False)


# In[ ]:





# In[ ]:




