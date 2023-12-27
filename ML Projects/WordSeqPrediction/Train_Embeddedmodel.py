import re
from keras.preprocessing.sequence import pad_sequences
import pickle

class TrainEmbeddedModel():
    def __init__(self,train_text):
        self.input_text=train_text

    def text_cleaner(self,text):
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

    def create_seq(self,text):
        length = 30
        sequence = list()
        for i in range(length, len(text)):
            # select sequence of token
            seq = text[i - length:i + 1]
            # store
            sequence.append(seq)
        print('Total sequence: %d ' % len(sequence))
        return sequence

    def mapping(self,text):
        chars = sorted(list(set(text)))
        mapping = dict((c, i) for i, c in enumerate(chars))
        return mapping

    def encode_sequ(self,seq,mapping):
        sequences = list()
        for line in seq:
            # integer encode line
            encoded_seq = [mapping[char] for char in line]
            # store
            sequences.append(encoded_seq)
        return sequences


    def buildModel(self,sequences,mapping):
        # create training and validation set
        from sklearn.model_selection import train_test_split
        import numpy as np
        from keras.utils import to_categorical
        from keras.preprocessing.sequence import pad_sequences
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, GRU, Embedding

        # vocalbulary size
        vocab = len(mapping)
        sequences = np.array(sequences)
        print('Sequences.......array')
        print(sequences.shape)

        # create X and Y
        X, y = sequences[:, :-1], sequences[:, -1]
        # one hot encode y
        y = to_categorical(y, num_classes=vocab)
        print(y)
        # create train and validation sets
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        print("Train shape : ", X_tr.shape, ' Val shape: ', X_val.shape)

        # Model building
        model = Sequential()
        model.add(Embedding(vocab, 50, input_length=30, trainable=True))
        model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
        model.add(Dense(vocab, activation='softmax'))
        print(model.summary())

        # complie the model
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
        # fit the model
        model.fit(X_tr, y_tr, epochs=50, verbose=2, validation_data=(X_val, y_val))
        return model

    def generate_seq(self, model, mapping, seq_length, seed_text, n_chars):
        in_text = seed_text
        print("in genereate seq")
        print(in_text)
        # generate a fixed number of characters
        for _ in range(n_chars):
            # encode the characters as integers
            encoded = [mapping[char] for char in in_text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict character
            print("Before predict. model......")
            print(model.summary())
            yhat = model.predict_classes(encoded, verbose=0)
            # reverse map integer to character
            out_char = ''
            for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # append to input
            in_text += char

        print('i,,,,,,,,,,,', in_text)
        return in_text


f = open('TrainData.txt', 'r+')
data_text = f.read()
modell=TrainEmbeddedModel(data_text)
clean_text=modell.text_cleaner(data_text)
seq_text=modell.create_seq(clean_text)
mapped_text=modell.mapping(clean_text)
print(mapped_text)
encode_seq=modell.encode_sequ(seq_text,mapped_text)
print("After encoded.......")
embedded_model=modell.buildModel(encode_seq,mapped_text)
#model is save
embedded_model.save('embedded.model')
print('model is successfully saved')


