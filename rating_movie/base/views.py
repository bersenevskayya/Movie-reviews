from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
stop_words = set(stopwords.words('english'))
model = load('./saved_models/model.joblib')
vect = load('./saved_models/tidf_vectorizerf.joblib')
def data_processing(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'^[\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

def predictor(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        text = data_processing(text)
        text_vector = vect.transform([text])
        prediction = np.round(model.predict(text_vector)[0][0])
        if prediction <=4:
            tonal = 'Негативный'
        elif prediction >=7:
            tonal = "Позитивный"
        return render(request, 'index.html', {'prediction': prediction, 'tonal': tonal})
    else:
        return render(request, 'index.html')

# Create your views here.
