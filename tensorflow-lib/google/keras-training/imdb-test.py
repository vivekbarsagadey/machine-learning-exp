from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print("train_data" ,train_data)
print("train_data[0]" ,train_data[0])
#print("train_labels" ,train_labels)
print("train_labels[0]" ,train_labels[0])

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



print("reverse_word_index > ",reverse_word_index)
print("reverse_word_index 'delivery' > ",reverse_word_index[65112])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print("decoded_review for train_data [0] > ",decoded_review)