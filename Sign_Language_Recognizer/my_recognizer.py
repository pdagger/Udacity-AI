import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    def item_probabilities(X):
        ''' Returns dictionnary of probabilities for items in test_set

        '''
        prob ={}
        max_logL = float('-inf')
        guess = ()
        for word, model in models.items():
            try:
                logL = model.score(X)            
            except:
                logL = float('-inf')
            prob[word] = logL
            if max_logL < logL:
              max_logL = logL
              guess = word

        return (prob, guess)

    Xlengths = test_set.get_all_Xlengths()
    for i in range(len(Xlengths)):
        # Xlengths dictionnary with items of the form {id: [X, length]}
        X = Xlengths[i][0]
        #length = Xlengths[i][0][1]
        prob, guess = item_probabilities(X)
        probabilities.append(prob)
        guesses.append(guess)  

    return (probabilities, guesses)
