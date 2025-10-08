# Step 1: Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Step 2: Sample training data
train_sentences = [
    ["I", "love", "NLP"],
    ["He", "plays", "football"],
    ["She", "writes", "code"]
]

train_tags = [
    ["PRON", "VERB", "NOUN"],
    ["PRON", "VERB", "NOUN"],
    ["PRON", "VERB", "NOUN"]
]

# Step 3: Build vocab and tag mappings
words = list(set(w for s in train_sentences for w in s))
tags = list(set(t for ts in train_tags for t in ts))

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Step 4: Convert to indices
X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in train_sentences]
y = [[tag2idx[t] for t in ts] for ts in train_tags]

# Step 5: Pad sequences
max_len = max(len(s) for s in train_sentences)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Step 6: One-hot encode labels
y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]

# Step 7: Build the model
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word2idx), output_dim=64, input_length=max_len)(input)
model = Bidirectional(LSTM(units=64, return_sequences=True))(model)
out = TimeDistributed(Dense(len(tag2idx), activation="softmax"))(model)

model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Step 8: Train the model
model.fit(X, np.array(y), batch_size=1, epochs=20)

# Step 9: Test cases
test_sentences = [
    ["I", "love", "NLP"],
    ["He", "plays", "football"],
    ["They", "run", "fast"]
]

test_tags = [
    ["PRON", "VERB", "NOUN"],
    ["PRON", "VERB", "NOUN"],
    ["PRON", "VERB", "ADV"]  # Expected tags
]

# Step 10: Predict and evaluate
print("\nSentence\t\tPredicted Tags\t\tCorrect (Y/N)")
for i, sentence in enumerate(test_sentences):
    x_test = [word2idx.get(w, word2idx["UNK"]) for w in sentence]
    x_test = pad_sequences([x_test], maxlen=max_len, padding='post')
    pred = model.predict(x_test)
    pred_tags = [idx2tag[np.argmax(p)] for p in pred[0][:len(sentence)]]

    expected_tags = test_tags[i]
    correct = "Y" if pred_tags == expected_tags else "N"

    print(f"{' '.join(sentence):<24}{' '.join(pred_tags):<24}{correct}")

Output :

<img width="636" height="277" alt="image" src="https://github.com/user-attachments/assets/d6ee1fd6-1975-48c6-82e7-98d0ac894162" />
<img width="671" height="561" alt="image" src="https://github.com/user-attachments/assets/2103cba9-6439-468d-9b1c-9ef46bdb5520" />
<img width="596" height="286" alt="image" src="https://github.com/user-attachments/assets/2540913f-4601-45f5-b936-c7066ada2fb2" />
