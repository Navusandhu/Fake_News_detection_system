# this will contain pre-processing step
import os
import re
import dill
import nltk
import joblib
from nltk.corpus import stopwords


class FakeNewsClassifier:
    def __init__(self, base_path):
        """
        TODO:
        1. load trained model
        2. load saved tokenizer
        """
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        # laoding bow convertor
        f = open(f"{base_path}/bow_convertor.pkl", 'rb')
        self.bow_transform = dill.load(f)

        # loading tfidf vectorizer
        vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
        self.text_vectorizer = joblib.load(
            vectorizer_path)

        # loading model
        model_path = os.path.join(base_path, 'tfidf_model.pkl')
        self.model = joblib.load(model_path)

        self.output = {0: "Reliable", 1: "Unreliable"}
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
        }

    def clean_text(self, text, remove_stopwords=True):

        # Convert words to lower case
        text = text.lower()

        # Replace contractions with their longer forms
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in self.contractions:
                    new_text.append(self.contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)

        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)

        # remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        # Tokenize each word
        text = nltk.WordPunctTokenizer().tokenize(text)

        return text

    def preprocess(self, x):
        """_summary_

        Parameters
        ----------
        x : str
            raw text
        """
        x = self.clean_text(x)
        bow = self.bow_transform.transform([x])
        sent_vector = self.text_vectorizer.transform(bow)
        return sent_vector   # np.array(sent_vector).reshape(1, -1)

    def postprocess(self, x):
        idx = x.item()
        return self.output[idx]

    def get_prediction(self, x):
        x = self.preprocess(x)
        output = self.model.predict(x)
        probab = self.model.predict_proba(x)
        return self.postprocess(output), probab
