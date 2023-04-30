import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re

class DataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizador=1):
        self.vectorizador = vectorizador
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        #Definicion de objetos para eliminar stopwords, eliminar texto sin sentido, realizar stemming y tokenizar
        stopword = stopwords.words('spanish')
        stemmer = SnowballStemmer('spanish')        
        for texto in X["review_es"]:
            resultado = []
            solo_letras = re.sub("[^a-zA-Záéíóúäëïöü]", " ", texto) 
            palabras = solo_letras.lower().split()
            for palabra in palabras:
                if palabra not in stopword:
                    palabra = stemmer.stem(palabra)
                    resultado.append(palabra)
            review = " ".join(resultado)
            X["review_es"] = list(map(lambda y: y.replace(texto, review), X["review_es"]))
        #vectorizador.fit(X["review_es"])
        tfidf_wm = self.vectorizador.transform(X["review_es"])
        X = pd.DataFrame(data = tfidf_wm.toarray(),index = X.index,columns = self.vectorizador.get_feature_names_out())
        return X
    