# Importing the libraries
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

#Getting trained tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function which cleans text
def text_cleaner(text_part):
    #removing bullet points
    text_part = re.sub("\*+\ ", "", text_part)
    #removing extra exclamations
    text_part = re.sub("!+", "!", text_part)
    #removing extra question marks
    text_part = re.sub("\?+", "?", text_part)
    #joining all different lines together
    text_part = re.sub("\n+", " ", text_part)
    #removing quotations
    text_part = re.sub("\"", "", text_part)
    #removing extra spaces
    text_part = re.sub("\ +", " ", text_part)
    text_part = text_part.lower()
    return text_part

# Inputting and cleaned the text
input_text = input('Enter your string: ')
processed_text = [text_cleaner(input_text)]

#sample_statement1 = ['The movie was absolutely fantastic to watch. Actors did a great job and kudos to the directors. Had a great time!']
#sample_statement2 = ['The movie was ok to watch. Was not that bad. Definetly a one time watch. Although there were a lot of points in the movie which could have made more appealing. Nonetheless, its worth a watch']
#sample_statement3 = ['This car looks absolutely disgusting. What a waste of money!!! And the performance is also not good, a car of this segment should deliver certain features which this model clearly lacks']

# tokenizing the text
tokenizer.fit_on_texts(processed_text)
sequence = tokenizer.texts_to_sequences(processed_text)
sequence = pad_sequences(sequence, maxlen=200)

# Model in action
model = load_model('trained_model.h5')
prediction = model.predict(sequence)
prediction_classes = ['Negative', 'Neutral', 'Positive']
predicted_class = prediction_classes[np.argmax(prediction)]

print('\n\nYour text has the sentiment: ', predicted_class)