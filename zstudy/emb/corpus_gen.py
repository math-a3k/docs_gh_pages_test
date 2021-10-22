import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from essential_generators import DocumentGenerator

def generate_corpus():
    """
    function to generate trainable data for model
    """
    gen = DocumentGenerator()
    lemmatizer = WordNetLemmatizer()
    unique_words = set([])
    unique_words_needed = 1000  # 5k

    # stop_words = []
    page = ""
    while len(unique_words) < unique_words_needed:
        sentence = gen.sentence()
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z]', ' ', sentence)   ### ascii only
        sentence = re.sub(r'\s+', ' ', sentence)  ### Removes all multiple whitespaces with a whitespace in a sentence
        sentence = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence) if word not in stopwords.words('english')]

        sentence = " ".join(sentence)
        sentence = re.sub(r'\b\w{1,2}\b', '', sentence)

        if len(sentence.split(' ')) < 5:
            continue

        page += sentence + "\n"
        
        for word in sentence.split(' '):
            unique_words.add(word)

    with open("data.cor", mode='w') as fp:        
        fp.write(page)

# code to train the model 
# gensim_model_train_save(dirinput="/home/vaibhav/Desktop/projects/freelance/japanese/myutil/utilmy/nlp/data.cor", dirout="./modelout/model.bin", pars={'min_count':2})



