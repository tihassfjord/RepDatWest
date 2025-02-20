import sys
import numpy as np
np.random.seed(1)

sys.path.append('code/')
sys.path.append('data/')
import gradient_checks
from HE_tweets_loader import load_tweets, vecToWords, load_tweets_concat
import rnn

dataset = 'shake'

print("******************\nSTARTING PROBLEM 2\n******************")
print("Gradient check for the fully connected layer implementations. Note, that an relative error of less than 1e-6\
most likely indicates that the gradient is correct.")

gradient_checks.check_gradient_fc()
gradient_checks.check_gradient_wordembedding()
gradient_checks.check_gradient_recurrent_step()
gradient_checks.check_gradient_rnn()

input("If the gradients look ok, press Enter to continue training the model...")

length = 20000
if dataset == 'shake':

    data = open('data/shake.txt', 'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    np.save('data/map_shake.npy', ix_to_char)

    p = 0
    seq_length = data_size
    
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    xs = np.zeros((vocab_size,data_size),dtype=np.int)
    
    for t in range(data_size):
        xs[inputs[t], t] = np.int(1)

    seq_length = 50
    model = rnn.RNN(vocab_size=vocab_size)
    model.train(np.array(inputs[:length]).reshape(1, length), 5001, 1, seq_length, dataset)

if dataset == 'trump':

    z = load_tweets_concat()
    z = z[z != 44]
    z = z[z != 43]
    
    x = z
    x = x[np.newaxis]
    seq_length = 50
    model = rnn.RNN()
    model.train(np.array(x[:, :length]), 6001, 1, seq_length, dataset)

