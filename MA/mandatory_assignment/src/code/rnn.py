#import layers_solution as layers
import layers
import networks
import numpy as np
from HE_tweets_loader import vecToWords

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def undo_grad_fc(fc_layer):
    dw = fc_layer.dw
    db = fc_layer.db
    fc_layer.w += 1e-2 * dw
    fc_layer.b += 1e-2 * db
    return fc_layer, dw, db
   
class RNN(networks.Network):
    def __init__(self, embedding_dim=256, dim_hid=256, vocab_size=45):
        """
        Implementation of a LSTM with dim_hid units.
        """
        self.vocab_size = vocab_size
        self.dim_hid = dim_hid
        self.embedding_dim = embedding_dim
        self.layers = [layers.WordEmbeddingLayer(vocab_size, embedding_dim), layers.LSTMLayer(embedding_dim, dim_hid), layers.FullyConnectedLayer(dim_hid, vocab_size), layers.SoftmaxLossLayer()]
        
        self.output = layers.SoftmaxLayer()
        self.store = None

        self.mwh = np.zeros_like(self.layers[1].wh)
        self.mwx = np.zeros_like(self.layers[1].wx)
        self.mb = np.zeros_like(self.layers[1].b)

        self.dx = None
        self.rnn_dwh = None
        self.rnn_dwx = None
        self.rnn_db = None
        self.fc_dw = None
        self.fc_db = None

    def forward_pass(self, x, h, c, TRAIN=False):
        """
        Forward pass of the RNN. If TRAIN=False the network outputs the softmax activations.
        If TRAIN=True the networks outputs the loss and the loss derivatives.
        """
        total_loss = 0
        outputs = []
        self.store = []
        self.fc_store = []
        self.N = x.shape[0]
        self.T = x.shape[1]
        self.e = self.layers[0].forward(x)
        h_prev = h
        c_prev = c
        fc = np.zeros((self.N, self.T, self.vocab_size))
        for i in range(self.T-1):
            h_new, c_new, cache = self.layers[1].forward_step(self.e[:, i, :], h_prev, c_prev)
            fc[:,i,:] += self.layers[2].forward(h_new)
            self.fc_store.append(self.layers[2].store)
            loss, loss_derivative = self.layers[3].forward(fc[:,i,:], x[:, i+1])
            total_loss += loss
            outputs.append(self.output.forward(fc[:,i,:]).argmax(1)[0])

            self.store.append([h_new, (h_new, self.e[:,i,:], h_prev, c_new, c_prev, cache), loss, loss_derivative])
            h_prev = h_new
            c_prev = c_new
        return fc, total_loss, h_new, c_new, outputs

    def backward_pass(self,fc=None, grad_check=False):
        """
        Backward pass of the RNN. Loops through timesteps starting from the last one.
        """
        
        self.dx = np.zeros((self.N, self.T, self.embedding_dim))
        self.rnn_dwh = np.zeros((self.layers[1].wh.shape))
        self.rnn_dwx = np.zeros((self.layers[1].wx.shape))
        self.rnn_db = np.zeros((self.layers[1].b.shape))
        self.fc_dw = np.zeros((self.layers[2].w.shape))
        self.fc_db = np.zeros((self.layers[2].b.shape))

        dh_prev = np.zeros((self.e.shape[0], self.dim_hid))
        dc_prev = np.zeros((self.e.shape[0], self.dim_hid))
    
        for i in reversed(range(self.T-1)):
            self.layers[2].store = self.fc_store[i]
            if grad_check == True:
                dfc = self.layers[2].backward(fc[:,i,:])
            else:
                dfc = self.layers[2].backward(self.store[i][3])
            self.fc_dw += self.layers[2].dw
            self.fc_db += self.layers[2].db            
            undo_grad_fc(self.layers[2]) #TODO Ugly hack to allow to use same FC implementation.
            dx_i, dh_i, dc_i, dwh_i, dwx_i, db_i = self.layers[1].backward_step(dfc + dh_prev, dc_prev, self.store[i][1])
            self.rnn_dwh += dwh_i
            self.rnn_dwx += dwx_i
            self.rnn_db += db_i
            self.dx[:, i, :] += dx_i
            dh_prev = dh_i
            dc_prev = dc_i

        # Clip gradients
        for dparam in [self.rnn_dwh, self.rnn_dwx, self.rnn_db, self.dx, self.fc_dw, self.fc_db]:
            np.clip(dparam, -5, 5, out=dparam)

        self.layers[0].backward(self.dx)

        self.mwh += self.rnn_dwh*self.rnn_dwh
        self.mwx += self.rnn_dwx*self.rnn_dwx
        self.mb += self.rnn_db*self.rnn_db
        self.layers[1].wh -= layers.update_param_adagrad(self.rnn_dwh, self.mwh)
        self.layers[1].wx -= layers.update_param_adagrad(self.rnn_dwx, self.mwx)
        self.layers[1].b -= layers.update_param_adagrad(self.rnn_db, self.mb)
        self.layers[2].w -= layers.update_param(self.fc_dw)
        self.layers[2].b -= layers.update_param(self.fc_db)

        

    def forward_pass_test(self, x, h, c):
        """
        Forward pass of the RNN. If TRAIN=False the network outputs the softmax activations.
        If TRAIN=True the networks outputs the loss and the loss derivatives.
        """
        T = 280
        self.N = x.shape[0]
        e = self.layers[0].forward(x)
        outputs = [x[0][0]]
        input_char = e[:,0,:]
        for i in range(T-1):
            h_new, c_new, _ = self.layers[1].forward_step(input_char, h, c)
            fc = self.layers[2].forward(h_new)
            p = self.output.forward(fc)[0]
            outputs.append(np.random.choice(range(self.vocab_size), p=p))
            h = h_new
            c = c_new
            input_char = self.layers[0].forward(outputs[-1])

        return outputs


    def train(self, x, epochs, batch_size, seq_length, dataset):
        """
        Training function. Creates batches of size
        batch_size and trains for epochs number of epochs.
        """

        idx_list = list(self.make_batches_rnn(x.shape[1], seq_length))
        for epoch in range(epochs):
            h = np.zeros((batch_size, self.dim_hid))
            c = np.zeros((batch_size, self.dim_hid))
            loss_av = []
            idx_to_test = np.random.randint(0, len(idx_list))
            for idx, val in enumerate(idx_list):
                fc, loss, h, c, outputs = self.forward_pass(x[:, val], h, c, True)
                self.backward_pass()
                loss_av.append(loss)
                if idx % len(idx_list) == idx_to_test and epoch % 1 == 0:
                    print("\n\n******************\n" + bcolors.FAIL + "EPOCH "+str(epoch)+bcolors.ENDC+"\n******************", flush=True)
                    print(bcolors.OKGREEN+"Loss: " + str(np.mean(loss_av)) + bcolors.ENDC, flush=True)
                    print('******************\n' + bcolors.WARNING + 'Prediction from input (just next char):' + bcolors.ENDC, flush=True)
                    vecToWords(np.array(outputs), dataset)
                    print('******************\n' + bcolors.WARNING + 'Generated character is fed as input to next time step:' + bcolors.ENDC, flush=True)
                    self.test(np.asarray([[outputs[0]]]), h, c, dataset)
                
    

    def test(self, x, h, c, dataset):
        """
        Testing network.
        """
        out = self.forward_pass_test(x, h, c)
        out = np.array(out)
        vecToWords(out, dataset)
        return None

