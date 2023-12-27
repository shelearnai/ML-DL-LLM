import re
from keras.preprocessing.sequence import pad_sequences

class EmbeddedModel():
    def __init__(self,input):
        self.input_text=input

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
        #print('Total sequence: %d ' % len(sequence))
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

    # generate a sequence of characters with a language model
    def generate_seq(self,model, mapping, seq_length, seed_text, n_chars):
        in_text = seed_text
        # generate a fixed number of characters
        for _ in range(n_chars):
            # encode the characters as integers
            encoded = [mapping[char] for char in in_text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict character
            yhat = model.predict_classes(encoded, verbose=0)
            # reverse map integer to character
            out_char = ''
            for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # append to input
            in_text += char
        return in_text

    def startProcess(self):
        import pickle
        f = open('TrainData.txt', 'r+')
        data_text = f.read()
        clean_text=self.text_cleaner(data_text)
        seq_text=self.create_seq(clean_text)
        mapped_text=self.mapping(clean_text)
        encode_seq=self.encode_sequ(seq_text,mapped_text)
        import tensorflow as tf
        model = tf.keras.models.load_model("embedded.model")
        r_data = self.generate_seq(model, mapped_text, 30, self.input_text.lower(), 15)
        return r_data