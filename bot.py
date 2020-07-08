import os
import discord
from dotenv import load_dotenv
from discord.ext import commands

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

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

#Getting trained tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('trained_model.h5')

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

def prediction_maker(input_text):
    processed_text = [text_cleaner(input_text)]
    # tokenizing the text
    tokenizer.fit_on_texts(processed_text)
    sequence = tokenizer.texts_to_sequences(processed_text)
    sequence = pad_sequences(sequence, maxlen=200)

    # Model in action
    prediction = model.predict(sequence)
    prediction_classes = ['Negative', 'Neutral', 'Positive']
    predicted_class = prediction_classes[np.argmax(prediction)]
    response = str("The sentiment of this comment is: " + predicted_class)
    return response

bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    guild = discord.utils.find(lambda g: g.name == GUILD, bot.guilds)
    print(f'{bot.user} has connected to Discord!')
    print(f'Guild name: {guild.name}(id: {guild.id})')

@bot.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(f'Hi {member.name}! Welcome to the Ripe Tomatoes Server! My name is Tomatina and my job is to predict the sentiment of sentences you enter in this chat. Type !help to know more')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.content == '!help':
        help_message = 'This bot is designed to classify comment sentiments on the fly. Just type in a comment and it will classify it for you as either positive, neutral or negative. A little detailed comments are encouraged for the bot to make a sound prediction. Please refrain from the use of emojis'
        response = help_message
    else:
        response = prediction_maker(input_text=message.content)
    await message.channel.send(response)

bot.run(TOKEN)