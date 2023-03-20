import numpy as np
from keras.datasets import imdb
from keras_preprocessing import sequence
from tensorflow import keras

# Uncomment if your machine is M1 and training the model is taking longer than expected
# tf.config.set_visible_devices([], 'GPU')

VOCAB_SIZE = 88584
MAX_LEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data, MAX_LEN)
test_data = sequence.pad_sequences(test_data, MAX_LEN)

# Uncomment first time to create the model, then you can just load it
# model = keras.Sequential([
#     keras.layers.Embedding(VOCAB_SIZE, 32),
#     keras.layers.LSTM(32),
#     keras.layers.Dense(1, activation="sigmoid")
# ])
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
# history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
# results = model.evaluate(test_data, test_labels)
# print(results)

# model.save('imdb_reviews')
model = keras.models.load_model('imdb_reviews')

word_index = imdb.get_word_index()


def encode_text(string):
    tokens = keras.preprocessing.text.text_to_word_sequence(string)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAX_LEN)[0]


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(f'That\'s a {"Positive" if result[0] >= 0.5 else "Negative"} Review!')


review = input("Write your opinion about the last movie you saw (the longer the better): ")
predict(review)
