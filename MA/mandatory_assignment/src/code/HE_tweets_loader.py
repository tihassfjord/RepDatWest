import numpy as np


def load_tweets():

    data = np.load('data/tweets.npy')

    x_tr = data[:6000]
    x_te = data[6000:]

    return x_tr, x_te


def load_tweets_concat():

    data = np.load('data/tweets.npy')

    x = data.ravel()

    return x


def vecToWords(data, dataset):

    if dataset == 'shake':
        # 20200213: np.load may fail for some versions of numpy
        # https://stackoverflow.com/questions/55824625/
        try:
            mapper = np.load('data/map_shake.npy').item()
        except:
            mapper = np.load('data/map_shake.npy', allow_pickle=True).item()
    if dataset == 'trump':
        try:
            mapper = np.load('data/map_back.npy').item()
        except:
            mapper = np.load('data/map_back.npy', allow_pickle=True).item()

    if len(data.shape) == 1:
        out = []
        run = True
        for i in data:
            if run:
                out.append(mapper[i])


        #print('Networks output:\n')
        print(''.join(out))
    else:
        for j in range(int(data.shape[0])):
            out = []
            run = True
            for i in data[j]:
                if run:
                    out.append(mapper[i])


            #print('Networks output:\n')
            print(''.join(out))

    pass
