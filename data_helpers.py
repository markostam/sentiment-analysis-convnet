import numpy as np
import re
import itertools
import codecs
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels_save_test_and_train():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Saves a test/train set for evaluation.
    Returns split sentences and labels.
    """
    # Load data from files and split into test/train
    positive_examples = list(codecs.open("./data/rt-polaritydata/rt-polarity.pos", "r", "utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    size = len(positive_examples)
    positive_train = positive_examples[:int(size*.8)]
    positive_test = positive_examples[int(size*.8):]

    negative_examples = list(codecs.open("./data/rt-polaritydata/rt-polarity.neg", "r", "utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    size = len(negative_examples)
    negative_train = negative_examples[:int(size*.8)]
    negative_test = negative_examples[int(size*.8):]
    #save held-out test set
    negative_test_file = codecs.open("./data/rt-polaritydata/test-rt-polarity.neg", 'w', "utf-8")
    negative_test_file.writelines(("%s \n" % item  for item in negative_test))
    positive_test_file = codecs.open("./data/rt-polaritydata/test-rt-polarity.pos", 'w', "utf-8")
    positive_test_file.writelines(("%s \n" % item  for item in positive_test))
    #save train/dev set
    negative_train_file = codecs.open("./data/rt-polaritydata/train-rt-polarity.neg", 'w', "utf-8")
    negative_train_file.writelines(("%s \n" % item  for item in negative_test))
    positive_train_file = codecs.open("./data/rt-polaritydata/train-rt-polarity.pos", 'w', "utf-8")
    positive_train_file.writelines(("%s \n" % item  for item in positive_test))
    # Split by words
    x_text = positive_train + negative_train
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_train]
    negative_labels = [[1, 0] for _ in negative_train]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_test_set():
    """
    Loads test MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files and split into test/train
    positive_examples = list(codecs.open("./data/rt-polaritydata/test-rt-polarity.pos", "r", "utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(codecs.open("./data/rt-polaritydata/test-rt-polarity.neg", "r", "utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
