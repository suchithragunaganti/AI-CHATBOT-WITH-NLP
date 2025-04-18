
import nltk
import numpy as np
import random
import string  # for processing text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Sample chatbot data
corpus = """
Hello, how can I help you?
What is your name?
I am a chatbot created using Python.
What do you do?
I can answer your basic questions.
Where do you live?
I live in your computer!
Goodbye
See you later
Thank you
You're welcome
"""

# Preprocess the corpus
sent_tokens = nltk.sent_tokenize(corpus)  # Convert into list of sentences
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "hey")
GREETING_RESPONSES = ["Hello!", "Hi there!", "Greetings!", "Hey! How can I assist you?"]

def greeting(sentence):
    """If user's input is a greeting, return a random greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generate response
def response(user_input):
    user_input = user_input.lower()
    sent_tokens.append(user_input)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    sent_tokens.pop()  # remove last input to avoid growing list

    if req_tfidf == 0:
        return "I'm sorry, I don't understand that."
    else:
        return sent_tokens[idx]

# Run chatbot loop
print("BOT: Hello! I am your assistant. Type 'bye' to exit.")

while True:
    user_input = input("YOU: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("BOT: Goodbye! Have a nice day.")
        break
    elif greeting(user_input) is not None:
        print("BOT:", greeting(user_input))
    else:
        print("BOT:", response(user_input))
