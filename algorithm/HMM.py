import numpy as np


class HiddenMarkovModel:
    def __init__(self, num_state=10):
        self.num_state = num_state  # number of hidden states

    def train(self, data, max_iter=100, epsilon=1e-2):
        self.build_dict(data)  # build one-hot
        self.start_dist = self.normalize(np.random.rand(self.num_state))  # initial distribution of hidden states
        self.transition = self.normalize(np.random.rand(self.num_state, self.num_state))  # prob of transfer between states
        self.emission = self.normalize(np.random.rand(self.num_state, self.num_observation))  # distribution of hidden states when observed

        epoch = 1
        while epoch <= max_iter:
            new_start_dist, new_transition, new_emission = self.Baum_Welch(data)
            move_start_dist = self.diff(new_start_dist, self.start_dist)
            move_transition = self.diff(new_transition, self.transition)
            move_emission = self.diff(new_emission, self.emission)
            print('Epoch {}: PI update: {}, Transition update: {}, Emission update: {}.'.format(epoch,move_start_dist,move_transition,move_emission))

            # Early stop.
            if move_start_dist < epsilon and move_transition < epsilon and move_emission < epsilon:
                break

            self.start_dist, self.transition, self.emission = new_start_dist, new_transition, new_emission
            # self.save('./data/saved_paras.npy')
            epoch += 1
        return self.start_dist, self.transition, self.emission, self.char_dict

    def build_dict(self, data):    # Build character dictionary using one-hot mapping.
        self.char_dict = []
        for i in range(data.shape[0]):
            sentence = data[i]
            new_sentence_set = set(sentence)
            self.char_dict = list(set(self.char_dict) | new_sentence_set)  # Remove redundancy.

        self.num_observation = len(self.char_dict)

    def normalize(self, mat):
        if len(mat.shape) == 1:
            return mat / mat.sum()
        return mat / mat.sum(axis=1).reshape(-1, 1)

    def Baum_Welch(self, data):
        # Forward (alpha) and Backward (beta).
        alpha_list, beta_list = [], []
        self.embed_data = []
        for sentence in data:
            sentence = self.embedding(sentence)
            self.embed_data.append(sentence)
            sen_len = sentence.shape[0]
            alpha, beta = np.zeros((sen_len, self.num_state)), np.zeros((sen_len, self.num_state))

            for ob_index in range(sen_len):
                for state in range(self.num_state):
                    if ob_index == 0:
                        alpha[ob_index, state] = self.emission[state, sentence[0]] * self.start_dist[state]
                        beta[sen_len-1-ob_index, state] = 1
                    else:
                        # Forward: compute previous observations and current state is i under current paras.
                        previous_sum = sum([alpha[ob_index - 1, previous] * self.transition[previous, state] \
                                            for previous in range(self.num_state)])
                        # Backward: compute probability of upcoming observations under current state and paras.
                        next_sum = sum([beta[sen_len-1-ob_index+1, next_] * self.transition[state, next_] * \
                                        self.emission[next_, sentence[sen_len-1-ob_index+1]] for next_ in range(self.num_state)])
                        alpha[ob_index, state] = previous_sum * self.emission[state, sentence[ob_index]]
                        beta[sen_len-1-ob_index, state] = next_sum
            alpha_list.append(self.normalize(alpha))
            beta_list.append(self.normalize(beta))

        # Expectation Maximization.
        gamma_list, xi_list = [], []

        # E-step.
        for index in range(data.shape[0]):
            sentence_e = self.embed_data[index]
            sen_len_e = len(sentence_e)
            gamma = np.zeros((sen_len_e, self.num_state))
            xi = np.zeros((sen_len_e - 1, self.num_state, self.num_state))
            # Gamma: probability of current state is i.
            for i in range(gamma.shape[0]):
                for state in range(self.num_state):
                    gamma[i, state] = alpha_list[index][i, state] * beta_list[index][i, state]
                gamma[i] /= gamma[i].sum()

            # Xi: probability of current state is i and next state is j.
            for i in range(xi.shape[0]):
                for previous in range(self.num_state):
                    for next_ in range(self.num_state):
                        xi[i, previous, next_] = alpha_list[index][i, previous] * self.emission[next_, sentence_e[i + 1]] * \
                                                        self.transition[previous, next_] * beta_list[index][i + 1, next_]
                xi[i] = xi[i] / xi[i].sum()

            gamma_list.append(gamma)
            xi_list.append(xi)

        # M_step. Update parameters using optimized solutions.
        new_start_dist, new_transition, new_emission = np.ones_like(self.start_dist), np.ones_like(self.transition), np.ones_like(self.emission)
        for state in range(self.num_state):
            new_start_dist[state] = sum([gamma_list[i][0, state] for i in range(data.shape[0])])/data.shape[0]

            for next_ in range(self.num_state):
                a_sum, b_sum = 0, 0
                for i in range(data.shape[0]):
                    for t in range(len(data[i]) - 1):
                        a_sum += xi_list[i][t, state, next_]
                        b_sum += gamma_list[i][t, state]
                new_transition[state, next_] = a_sum / b_sum

            for char in range(self.num_observation):
                c_sum, d_sum = 0, 0
                for i in range(data.shape[0]):
                    for t in range(len(data[i])):
                        if self.embed_data[i][t] == char:
                            c_sum += gamma_list[i][t, state]
                        d_sum += gamma_list[i][t, state]
                new_emission[state, char] = c_sum / d_sum

        return new_start_dist, new_transition, new_emission

    def embedding(self, sentence):  # Convert texts into numerical vectors.
        return np.array([self.char_dict.index(char) for char in sentence])

    def diff(self, a, b):  # Compute magnititude of update.
        return np.linalg.norm(a-b)

    def generate(self, n):
        gene_list = []
        state = np.random.randint(self.num_state)
        for i in range(n):
            word = np.random.choice(self.char_dict, p=self.emission[state])
            if word != '.':
                word += ' '
            gene_list.append(word)
            state = np.random.choice(range(self.num_state), p=self.transition[state])
        print("".join(gene_list))

    def predict(self, text, n):
        for item in text:
            if item not in self.char_dict:
                print('Invalid input: input not in Shakespeare\'s dictionary!')
                return
        state_list = self.viterbi(text)
        gene_list = []
        state = int(state_list[-1])
        for i in range(n):
            word = np.random.choice(self.char_dict, p=self.emission[state])
            if word != '.':
                word += ' '
            gene_list.append(word)
            state = np.random.choice(range(self.num_state), p=self.transition[state])
        print('Next word(s)', "".join(gene_list))

    def viterbi(self, text):
        p_list = np.zeros((len(text), self.num_state))
        state_list = np.zeros((len(text), self.num_state))

        for i in range(p_list.shape[0]):
            for state in range(self.num_state):
                if i == 0:
                    p_list[i, state] = self.start_dist[state] * self.emission[state, self.char_dict.index(text[i])]
                    state_list[i, state] = 0
                else:
                    p = [p_list[i-1, j] * self.transition[j, state] * \
                         self.emission[state, self.char_dict.index(text[i])] for j in range(self.num_state)]
                    p_list[i, state] = max(p)
                    state_list[i, state] = p.index(max(p))

        best_path = np.zeros(p_list.shape[0])
        best_path[-1] = int(np.argmax(p_list[-1]))
        for t in range(p_list.shape[0]-1):
            # print(best_path)
            best_path[p_list.shape[0]-2-t] = state_list[p_list.shape[0]-1-t, int(best_path[p_list.shape[0]-1-t])]
        return best_path

    def load_paras(self, paras_file):
        paras = np.load(paras_file, allow_pickle=True)
        self.start_dist = paras[:, 0].reshape(-1)
        self.transition = paras[:, 1:self.num_state+1]
        self.emission = paras[:, self.num_state+1:]

    def save(self, direct):
        a = np.hstack((self.start_dist.reshape(-1,1), self.transition, self.emission))
        np.save(direct, a)




