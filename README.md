# EECS738 Project2 Hidden Markov Model
EECS738 Machine Learning course project2. Hidden Markov Model is implemented with three principle functions: training, text generation and word prediction. More specifically, Baum Welch Algorithm is used to train the model without hidden states given, and Viterbi Algorithm is used to search for optimal hidden states and predict words.

## Dataset
* [Shakespeare Plays.](https://www.kaggle.com/kingburrito666/shakespeare-plays) This dataset contains all Shakespeare's plays. The raw text file is used.

If you'd like to use your own data, please put them under ['data'](https://github.com/liuzey/EECS738_Project2/tree/main/data) and input the filename for training/testing. Pay attention to data preprocessing, e.g. [punctuation removal](https://github.com/liuzey/EECS738_Project2/blob/d5351c0f37cb108852f6ae24dc9216af31a128a7/main.py#L25). In my implementation, I keep the full stop marks as I think they are essential in identifying semantic endings.


## Ideas and Theoretical Basis
* **Training**: Using text only, this is an unsupervised learning problem which can be solved by Baum Welch Algorithm with forward algorithm and backward algorithm, which is similar to Expectation-Maximization algorithm. For every current hidden state, the probability of its every possible state based on previous observations and the prior probability of upcoming observations are calculated, which combines to estimate the probability of 'certainly happen' data under the model setting. Thus, the model parameters can be tuned accrodingly to fit the data (text), which learns 'grammar'.
* **Generation**: Using random seed as the intial state, text can be generated by transfering to states and picking out observations based on learned distributions.
* **Prediction**: Following a similar pipeline as training, the probability of current state can be estimated when model parameters are given. Thus, the best state sequence can be drawn for a certain text, and next words can be predicted using state transfer.

## Setup
### Environment
* Python 3.8
* MacOS or Linux

### Package
* Recommend setting up a virtual environment. Different versions of packages may cause unexpected failures.
* Install packages in **requirements.txt**.
```bash
pip install -r requirements.txt
``` 
* Note that NLTK package (version==3.5) is used for tokenized sentences.

## Usage
### Positional & Optional Parameters
* **data**: Data filename, e.g. 'alllines.txt'.
* **-t**: Task to perform. 0 for word prediction, and 1 for text generation. (Default: -1).
* **-n**: Number of hidden states. (Default: 10).
* **-f**: If None, model will be trained from scratch. Otherwisse, model parameters are loaded using this filename. (Default: ''.).

### Example
```bash
python main.py alllines.txt -t 1 -n 10
```
* Number of hidden state is ten.
* Model is trained from scratch.
* Perform text generation.
```bash
python main.py alllines.txt -t 0 -n 10 -f 'saved_paras.npy'
```
* Number of hidden state is ten.
* Model parameters are pre-trained and loaded from './data/saved_paras.npy'.
* Perform word prediction. For word prediction, a text must be given in advance when command teminal requests.

## Results
### Text Generation
Here are some text generated by trained HMM:
* better plays wedded improvident unborn hose man herd know'st unborn 
* agues main top silly unworthy vilely paris argue signior let 
* destruction snatches single enforced engenders represent wildfire adam tide heavens 
* sighing sex partlet seldom dastards knave whelps partlet quick sawest 
* audience exercise cranking compassion mantle driveth bars birth stuck c 

### Word Prediction
Here are some predicted words by trained HMM: (Default 2 words to come)
* **Input**: We are on the, **Next word(s)**: witches odds 
* **Input**: There are his, **Next word(s)**: talkest glansdale 
* **Input**: I'll go, **Next word(s)**: ransom vaward 
* **Input**: How shall, **Next word(s)**: foully piece
* **Input**: Case of, **Next word(s)**: thighs grease 



## Notes
* Default iteration limit is set at 100. If the update between subsequent steps are too small, the model will be believed to converge and iterations will stop early.
* The training process can be very time-consuming. Only the first 6,000 words are used for training. As the text data increases, both the batches to iterate and the size of dictionary, i.e. one dimension of emission matrix, increase, which will exponentially add to time complexity. If any change is needed in this setting， please refer to [here](https://github.com/liuzey/EECS738_Project2/blob/af0c252da06b6a10647e082f06b02b121b6abdfc/main.py#L38)
* Overall, the performance of HMM over Shakespeare plays is not satisfying. This can be due to different causes: embedding strategies, dataset size for training and model structrue, e.g. LSTM.

## Schedule
- [x] Set up a new git repository in your GitHub account
- [x] Pick a text corpus dataset such as https://www.kaggle.com/kingburrito666/shakespeare-plays or from https://github.com/niderhoff/nlp-datasets.
- [x] Choose a programming language (Python, C/C++, Java). **Python**
- [x] Formulate ideas on how machine learning can be used to model distributions within the dataset.
- [x] Build a Hidden Markov Model to be able to programmatically: 1.Generate new text from the text corpus. 2.Perform text prediction given a sequence of words.
- [x] Document your process and results.
- [x] Commit your source code, documentation and other supporting files to the git repository in GitHub.

## Reference
* Baum–Welch algorithm - Wikipedia. https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
* Hidden Markov Models - scikit-learn. https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.hmm.GaussianHMM.html
* hmmlearn - Github. https://github.com/hmmlearn/hmmlearn
* https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
* https://www.fing.edu.uy/~alopeza/biohpc/papers/hmm/Eddy-What-is-a-HMM-ATG4-preprint.pdf
* nltk.tokenize package - NLTK 3.5 documentation. https://www.nltk.org/api/nltk.tokenize.html
* Random sampling - Numpy Api Reference. https://numpy.org/doc/stable/reference/random/

