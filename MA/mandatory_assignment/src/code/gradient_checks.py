#import layers_solution as layers
import layers
import utils
import copy
import numpy as np
import rnn

def replaceWeight(x, layer, Wx, b):
    layer.w = Wx
    layer.b = b
    dx = layer.forward(x)
    return dx

def check_gradient_conv():
    F = 10
    N = 5
    filtersize = (20, 2, 3, 3)
    layer = layers.ConvolutionalLayer(filtersize)
    x = np.random.randn(N, filtersize[1], F, F)
    Wx = np.random.normal(0, 0.001, filtersize)
    b = np.random.normal(0, 0.001, (filtersize[0],))
    
    layer.w = Wx
    layer.b = b
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    
    fx_x = lambda x: layer.forward(x)
    fW_x = lambda Wx: replaceWeight(x, layer, Wx, b)
    fb_x = lambda b: replaceWeight(x, layer, Wx, b)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_x, x, dnext_x)
    dWx_num = num_grad(fW_x, Wx, dnext_x)
    
    db_num = num_grad(fb_x, b, dnext_x)
    
    dx = layer.backward(dnext_x)
    dWx = layer.dw
    db = layer.db
    
    print("Relative error for convolutional layer:")
    print("dx: ", utils.rel_error(dx_num, dx))
    print("dWx: ", utils.rel_error(dWx_num, dWx))
    print("db: ", utils.rel_error(db_num, db))

def check_gradient_fc():
    F = 10
    N = 5
    dim_in = 4
    dim_out = 6
    layer = layers.FullyConnectedLayer(dim_in, dim_out)
    x = np.random.randn(N, dim_in)
    Wx = np.random.normal(0, 0.001, (dim_in, dim_out))
    b = np.random.normal(0, 0.001, (dim_out,))
    
    layer.w = Wx
    layer.b = b
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    
    fx_x = lambda x: layer.forward(x)
    fW_x = lambda Wx: replaceWeight(x, layer, Wx, b)
    fb_x = lambda b: replaceWeight(x, layer, Wx, b)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_x, x, dnext_x)
    dWx_num = num_grad(fW_x, Wx, dnext_x)
    
    db_num = num_grad(fb_x, b, dnext_x)
    
    dx = layer.backward(dnext_x)
    dWx = layer.dw
    db = layer.db
    
    print("Relative error for fully connected layer:")
    print("dx: ", utils.rel_error(dx_num, dx))
    print("dWx: ", utils.rel_error(dWx_num, dWx))
    print("db: ", utils.rel_error(db_num, db))

def check_gradient_pool():
    F = 28
    N = 4
    C = 3
    pool_r, pool_c, stride = 2, 2, 2
    layer = layers.MaxPoolingLayer(pool_r, pool_c, stride)
    x = np.random.randn(N, C, F, F)
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    
    fx_x = lambda x: layer.forward(x)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_x, x, dnext_x)
    
    dx = layer.backward(dnext_x)
    
    print("Relative error for pooling layer:")
    print("dx: ", utils.rel_error(dx_num, dx))
    
def check_gradient_relu():
    F = 28
    N = 4
    C = 3
    layer = layers.ReluLayer()
    x = np.random.randn(N, C, F, F)
    x = np.abs(x)+1
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    dnext_x = np.abs(dnext_x)+1
    
    fx_x = lambda x: layer.forward(x)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_x, x, dnext_x)
    
    dx = layer.backward(dnext_x)
    
    dx_pos = utils.rel_error(dx_num, dx)
    
    x = -np.abs(x)-1
    dnext_x = -np.abs(dnext_x)-1
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)-1
    
    dx_num = num_grad(fx_x, x, dnext_x)
    
    dx = layer.backward(dnext_x)
    print("Relative error for ReLU layer:")
    print("dx: ", (dx_pos + utils.rel_error(dx_num, dx))/2.0)

def check_gradient_tanh():
    F = 28
    N = 4
    C = 3
    layer = layers.TanhLayer()
    x = np.random.randn(N, C, F, F)
    
    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    
    fx_x = lambda x: layer.forward(x)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_x, x, dnext_x)
    
    dx = layer.backward(dnext_x)
    
    print("Relative error for TanH layer:")
    print("dx: ", utils.rel_error(dx_num, dx))

def replaceWeightWE(x, layer, w):
    layer.w = w
    dx = layer.forward(x)
    return dx

def check_gradient_wordembedding():
    T = 4
    N = 6
    embedding_dim = 5
    vocab_dim = 8
    layer = layers.WordEmbeddingLayer(vocab_dim, embedding_dim)
    x = np.random.randint(0,vocab_dim,(N, T))
    w = np.random.normal(0, 0.1, (vocab_dim, embedding_dim))
    
    layer.w = w    

    next_x = layer.forward(x)
    dnext_x = np.random.randn(*next_x.shape)
    
    fw_x = lambda w: replaceWeightWE(x, layer, w)
    
    num_grad = utils.eval_numerical_gradient_array
    
    dw_num = num_grad(fw_x, w, dnext_x)
    
    layer.backward(dnext_x)
    dw = layer.dw

    print("Relative error for WordEmbedding layer:")
    print("dw: ", utils.rel_error(dw_num, dw))

def replaceWeightRecurrent(x,  h, c, layer, wx, wh, b):
    layer.wx = wx
    layer.wh = wh
    layer.wb = b
    dx = layer.forward_step(x, h, c)
    return dx

def check_gradient_recurrent_step():

    N = 4
    dim_in = 3
    dim_hid = 6
    layer = layers.LSTMLayer(dim_in, dim_hid)
    x = np.random.randn(N, dim_in)
    h = np.random.randn(N, dim_hid)
    c = np.random.randn(N, dim_hid)
    wx = np.random.normal(0, 0.1, (dim_in, 4*dim_hid))
    wh = np.random.normal(0, 0.1, (dim_hid, 4*dim_hid))
    b = np.random.normal(0, 0.1, (4*dim_hid,))
    
    layer.wx = wx
    layer.wh = wh
    layer.b = b

    next_h, next_c, cache = layer.forward_step(x, h, c)
    dnext_h = np.random.randn(*next_h.shape)
    dnext_c = np.random.randn(*next_c.shape)
    
    fx_h = lambda x: layer.forward_step(x,h,c)[0]
    fh_h = lambda h: layer.forward_step(x,h,c)[0]
    fc_h = lambda c: layer.forward_step(x, h, c)[0]
    fwx_h = lambda wx: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[0]
    fwh_h = lambda wh: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[0]
    fb_h = lambda b: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[0]

    fx_c = lambda x: layer.forward_step(x, h, c)[1]
    fh_c = lambda h: layer.forward_step(x, h, c)[1]
    fc_c = lambda c: layer.forward_step(x, h, c)[1]
    fwx_c = lambda wx: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[1]
    fwh_c = lambda wh: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[1]
    fb_c = lambda b: replaceWeightRecurrent(x, h, c, layer, wx, wh, b)[1]
    
    num_grad = utils.eval_numerical_gradient_array
    
    dx_num = num_grad(fx_h, x, dnext_h) + num_grad(fx_c, x, dnext_c)
    dh_num = num_grad(fh_h, h, dnext_h) + num_grad(fh_c, h, dnext_c)
    dc_num = num_grad(fc_h, c, dnext_h) + num_grad(fc_c, c, dnext_c)
    dwx_num = num_grad(fwx_h, wx, dnext_h) + num_grad(fwx_c, wx, dnext_c)
    dwh_num = num_grad(fwh_h, wh, dnext_h) + num_grad(fwh_c, wh, dnext_c)
    db_num = num_grad(fb_h, b, dnext_h) + num_grad(fb_c, b, dnext_c)

    dx, dh, dc, dwh, dwx, db = layer.backward_step(dnext_h, dnext_c, (next_h, x, h, next_c, c, cache))

    print("Relative error for RecurrentLayer layer:")
    print("dx: ", utils.rel_error(dx_num, dx))
    print("dh: ", utils.rel_error(dh_num, dh))
    print("dc: ", utils.rel_error(dc_num, dc))
    print("dwx: ", utils.rel_error(dwx_num, dwx))
    print("dwh: ", utils.rel_error(dwh_num, dwh))
    print("db: ", utils.rel_error(db_num, db))
    
def replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b):
    layer.layers[0].w = word_w
    layer.layers[1].wx = rnn_wx
    layer.layers[1].wh = rnn_wh
    layer.layers[1].b = rnn_b
    layer.layers[2].w = fc_w
    layer.layers[2].b = fc_b
    loss = layer.forward_pass(x, h, c)[0]
    return loss

def check_gradient_rnn():
    T = 3
    N = 5
    dim_hid = 4
    vocab_size = 2
    embedding_dim = 7
    layer = rnn.RNN(embedding_dim, dim_hid, vocab_size)
    x = np.random.randint(0,vocab_size,(N, T))
    h = np.random.randn(N, dim_hid)
    c = np.random.randn(N, dim_hid)
    
    rnn_wx = np.random.normal(0, 0.1, (embedding_dim, 4*dim_hid))
    rnn_wh = np.random.normal(0, 0.1, (dim_hid, 4*dim_hid))
    rnn_b = np.random.normal(0, 0.1, (4*dim_hid,))
    fc_w = np.random.normal(0, 0.1, (dim_hid, vocab_size))
    fc_b = np.random.normal(0, 0.1, (vocab_size,))

    word_w = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    next_x = layer.forward_pass(x, h, c)[0]
    dnext_x = np.random.randn(*next_x.shape)
    
    f_we = lambda word_w: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    f_rnn_wx = lambda rnn_wx: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    f_rnn_wh = lambda rnn_wh: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    f_rnn_b = lambda rnn_b: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    f_fc_w = lambda fc_w: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    f_fc_b = lambda fc_b: replaceWeightRnn(x, h, c, layer, word_w, rnn_wx, rnn_wh, rnn_b, fc_w, fc_b)
    

    num_grad = utils.eval_numerical_gradient_array
    
    dwe_num = num_grad(f_we, word_w, dnext_x) 
    dwx_num = num_grad(f_rnn_wx, rnn_wx, dnext_x)
    dwh_num = num_grad(f_rnn_wh, rnn_wh, dnext_x)
    db_num = num_grad(f_rnn_b, rnn_b, dnext_x) 
    dfw_num = num_grad(f_fc_w, fc_w, dnext_x)
    dfb_num = num_grad(f_fc_b, fc_b, dnext_x)  
	
    layer.backward_pass(dnext_x,grad_check=True)
    we_dw = layer.layers[0].dw
    dwh = layer.rnn_dwh
    dwx = layer.rnn_dwx
    db = layer.rnn_db    
    fc_dw = layer.fc_dw
    fc_db = layer.fc_db

    print("Relative error for full RNN:")
    print("dw_word: ", utils.rel_error(dwe_num, we_dw))
    print("dwx: ", utils.rel_error(dwx_num, dwx))
    print("dwh: ", utils.rel_error(dwh_num, dwh))
    print("db: ", utils.rel_error(db_num, db))
    print("fc_w: ", utils.rel_error(dfw_num, fc_dw))
    print("fc_b: ", utils.rel_error(dfb_num, fc_db))

