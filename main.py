# This is for Project2 - Hidden Markov Model in course EECS738 Machine Learning.
# Written by Zeyan Liu (StudentID: 3001190).
# Run command example:
# {python main.py alllines.txt -t 0 -n 10 -f 'saved_paras.npy'} for text prediction using pre-trained paras.
# {python main.py alllines.txt -t 1 -n 10} for text generation after training from scratch.

import os
import sys
import time
import numpy as np
import argparse
import nltk
from algorithm import HMM

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help='Text file for training.')
parser.add_argument("-t", type=int, default=-1, help='Task to do: {0.Prediction, 1.Generation}.')
parser.add_argument("-n", type=int, default=10, help='Number of hidden states for HMM.')
parser.add_argument("-f", type=str, default="", help='Pre-trained parameters. If not specified, train from scratch.')
args = parser.parse_args()

nltk.download('punkt')
DATA_DIR = './data/' + args.data
PARA_DIR = './data/' + args.f
punc = ['$', '6', '!', '0', "'", '5', ']', "''", '``', '3', ',', '1', '2', '-', '?', '7', ' ', '4', '8', ':', '9', ')',
        '(', '\t', '[']
if not os.path.exists(DATA_DIR):
    raise Exception('Invalid Directory.')


def load_data():
    data = []
    with open('./data/alllines.txt') as f:
        for line in f.readlines():
            line_t = [word.lower() for word in nltk.word_tokenize(line)]
            line_t = [item for item in line_t if item not in punc]
            data.append(line_t)
    return np.array(data[:20000])


def predict():
    # given = 'We are on the'
    given = input('Please input a text: \n')
    given_t = [word.lower() for word in nltk.word_tokenize(given)]
    given_t = [item for item in given_t if item not in punc]
    model.predict(given_t, 2)  # Predict the next two words.


def generate():
    model.generate(10)  # Generate a text with ten words.


if __name__ == '__main__':
    data = load_data()
    print('Data successfully loaded.')
    model = HMM.HiddenMarkovModel(num_state=args.n)
    start = time.time()
    if args.f:
        print('\n-----Pretrained-----')
        model.build_dict(data)
        model.load_paras(PARA_DIR)
    else:
        print('\n-----Training Begins-----')
        model.train(data, max_iter=100)

    if args.t == 0:
        print('\n-----Prediction-----')
        predict()
    elif args.t == 1:
        print('\n-----Generation-----')
        generate()

    model.save('./data/saved_paras.npy')
    print('\n-----Time Used: {}s-----'.format(time.time()-start))
