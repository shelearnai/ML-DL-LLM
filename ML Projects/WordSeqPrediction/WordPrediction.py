import sys

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import  QImage,QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QDir, Qt, QUrl
import time
import random
import numpy as np
import pandas as pd
import re
from EmbeddedModelPrediction import EmbeddedModel

class WordPrediction(QDialog):

    def __init__(self):
        super(WordPrediction,self).__init__()
        loadUi('wordPrediction.ui',self)
        self.image=None
        self.processedImage=None
        self.pushButton.clicked.connect(self.ngramPrediction)
        self.pushButton_2.clicked.connect(self.RNNmodelPrediction)

    def text_cleaner(text):
            global data_text
            # lower case text
            newString = text.lower()
            newString = re.sub(r"'s\b", "", newString)

            # remove punctuations
            newString = re.sub("[^a-zA-Z]", " ", newString)
            long_words = []

            # remove short word
            for i in newString.split():
                if len(i) >= 3:
                    long_words.append(i)
            return (" ".join(long_words)).strip()


    @pyqtSlot()
    def RNNmodelPrediction(self):
        input_text = self.textEdit.toPlainText()
        emodel=EmbeddedModel(input_text)
        predicted_word=emodel.startProcess()
        self.textEdit_2.setText(predicted_word)

    @pyqtSlot()
    def ngramPrediction(self):
        input_text = self.textEdit.toPlainText()
        n_gram_predicted_words = self.N_gramprediction(input_text)
        print(n_gram_predicted_words)
        self.textEdit_2.setText(n_gram_predicted_words)

    def N_gramprediction(self,input_text):
        from nltk.corpus import reuters
        from nltk import bigrams, trigrams
        from collections import Counter, defaultdict
        import random

        # Create a placeholder for model
        model = defaultdict(lambda: defaultdict(lambda: 0))
        # Count frequency of co-occurance
        for sentence in reuters.sents():
            for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                model[(w1, w2)][w3] += 1

        # Let's transform the counts to probabilities
        for w1_w2 in model:
            total_count = float(sum(model[w1_w2].values()))
            for w3 in model[w1_w2]:
                model[w1_w2][w3] /= total_count

        # starting word
        text1 = str(input_text)
        text = list(text1.split(' '))
        sentence_finished = False
        print(text1)
        while not sentence_finished:
            # select a random probability threshold
            r = random.random()
            accumulator = .0

            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break

            if text[-2:] == [None, None]:
                sentence_finished = True

        return (' '.join([t for t in text if t]))

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = WordPrediction()
    window.setWindowTitle('Remaining Word Prediction............')
    window.show()
    sys.exit(app.exec_())