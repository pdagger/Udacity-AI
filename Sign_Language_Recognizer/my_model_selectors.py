import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        n_samples = np.shape(self.X)[0]
        logN = math.log(n_samples)
        n_features = np.shape(self.X)[1]
        bics = [] # stores tuple of BIC score with corresponding num_states and model
        for num_states in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_states)
            try:
                logL = model.score(self.X, self.lengths)
                # Number of parameters
                p = num_states**2 + 2 * num_states * n_features - 1
                bic = -2 * logL + p * logN
                bics.append((bic, num_states, model))
            except: continue    

        # Probability between 0 and 1, then logL between -inf and 0 
        # logL is monotonic increasing function
        # BIC proportional to -logL
        # Then best BIC corresponds to min(bics)
        try:
            best_bic = min(bics)
            hmm_model = best_bic[2]
        except:
            # If best_dic empty take n_components=5  (what is prefered in the video lectures) 
            hmm_model = self.base_model(5)       

        return hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1) * SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        def anti_likelihood_sum(model, logL):
            ''' Generates the sum of the other words' likelihood to be generated by model without including
                the likelihood of the word fo wich the model was generated

            '''
            anti_logL = 0
            for word, data in self.hwords.items():
                # data[0] and data[1] are X and lenghts for corresponding 'word'
                try:
                    anti_logL = model.score(data[0], data[1])
                except: continue
            
            # anti_logL does not include the likelihood of the word that created the model
            anti_logL = anti_logL - logL
            return anti_logL

        dics =[] # stores tuple of DIC score with corresponding num_states and model
        for num_states in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_states)
            try:
                logL = model.score(self.X, self.lengths)
                dic = logL - anti_likelihood_sum(model, logL) / (len(self.hwords) - 1)
                dics.append((dic, num_states, model))
            except: continue

        # Get higher DIC score
        try:
            best_dic = max(dics)
            hmm_model = best_dic[2]
        except:
            # If best_dic empty take n_components=5  (what is prefered in the video lectures) 
            hmm_model = self.base_model(5)  

        return hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        n_splits = min(5, len(self.sequences))
        # print("{}-fold Cross-Validation for {}:".format(n_splits, self.this_word))

        def CV_test(num_states):
            '''Returns a tuple with the average score logL obatined by cross-validation 
               with the corresponding number of states to obtain it 

            '''
            split_method = KFold(n_splits=n_splits)
            logL = 0
            i = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # Get CV training sets
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                # Train
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                # Get CV testing sets
                test_X, test_lengths = combine_sequences(cv_train_idx, self.sequences)
                logL += model.score(test_X, test_lengths)
                i += 1
            return (logL / i, num_states)

        # Running CV for all possible values of n_components to store scores with corresponding n_components in a list
        cv_logL = []
        for num_states in range(self.min_n_components, self.max_n_components+1):
            # If can't perform CV in sequence because too small, take base model
            if len(self.sequences) < 2:
                model = self.base_model(num_states)
                try:
                    logL = model.score(self.X, self.lengths)
                except:
                    logL = float('-inf')
                cv_logL.append((logL, num_states, model)) # Also store model to avoid replicating operations
            else:
                try:
                    cv_logL.append(CV_test(num_states))
                except:
                    continue
                    #print('Value Error')

        # Choosing num_states for best logL
        try:
            optimal_num_states = max(cv_logL)[1]
        except:
            # If best_dic empty take n_components=5  (what is prefered in the video lectures) 
            optimal_num_states = 5

        if len(self.sequences) < 2:
            hmm_model = max(cv_logL)[2]
        else:
            hmm_model = GaussianHMM(n_components=optimal_num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
   
        # print("    Optimal number of components for {}: {}".format(self.this_word, hmm_model.n_components))

        return hmm_model