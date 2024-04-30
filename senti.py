# Define a route for sentiment prediction uvicorn essai2:app --reload
from fastapi import FastAPI
import os
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re, nltk
nltk.download("stopwords")
from joblib import load
# Create a FastAPI instance
app = FastAPI()

model = load('reg_model.joblib')



class InputData(BaseModel):
    text: str


# Define a route for sentiment prediction
@app.post("/predict/")
def predict_sentiment(input_data: InputData):
    text = input_data.text

    text = cleaned_text(text)
    sentiment = model.trained_model.predict(text)[0]
    sentiment_label = "Negative" if sentiment == 0 else "Positive"

    return {"sentiment": sentiment_label}

def cleaned_text(text):
    clean = re.sub("\n"," ",text)
    clean=clean.lower()
    clean=re.sub(r"[~.,%/:;?_&+*=!-]"," ",clean)
    clean=re.sub("[^a-z]"," ",clean)
    clean=clean.lstrip()
    clean=re.sub("\s{2,}"," ",clean)
    clean = word_tokenize(clean)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)    
    return cleaned_text