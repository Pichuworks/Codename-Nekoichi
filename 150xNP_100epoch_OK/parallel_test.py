'''
fuck rna
fuck deep learning
'''
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Model
from keras.layers import Input, Dropout, Embedding, LSTM 
from keras.layers import Activation, dot, TimeDistributed
from keras.layers import concatenate, Dense, Bidirectional
from keras.models import model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from time import time

str_sequence = list()
str_structure = list()

# 测试集做测试
test_str_seq = list()
test_str_str = list()

with open('sequence_150_np.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        str_sequence.append(str(line))

with open('structure_150_np.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        str_structure.append(str(line))

with open('test_set.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        test_str_seq.append(str(line))

with open('test_ans.txt') as f:
    for line in f.readlines():
        line = line.strip() 
        test_str_str.append(str(line))

# create sequence & structure data matrix

list_sequence = list()
list_structure = list()
list_test_seq = list()
list_test_str = list()

for word in str_sequence:
    list_sequence.append(word.split(' '))

for word in str_structure:
    list_structure.append(word.split(' '))

for word in test_str_seq:
    list_test_seq.append(word.split(' '))

for word in test_str_str:
    list_test_str.append(word.split(' '))

# get the number of unique tokens in each language
# print('SEQ: total unique tokens:', len(set([word for sent in list_sequence for word in sent])))
# print('SEQ: total unique tokens:', len(set([word for sent in list_structure for word in sent])))

# get the maximum input lengths for each language
# assuming normal distribution, mean + 2 stds = 97.8% of lengths
# add one for SOS tag
fuck_var = max([len(s)+1 for s in list_sequence])
maxlen = fuck_var

# print('SEQ: maximum string len :', max([len(s)+1 for s in list_sequence]))
# print('SEQ: avg len + 2 stdevs :', np.mean([len(s)+1 for s in list_sequence]) + 2 * np.std([len(s) for s in list_sequence]))
# print()
# print('STR: avg len + 2 stdevs :', np.mean([len(s)+1 for s in list_structure]) + 2 * np.std([len(s) for s in list_structure]))
# print('STR: maximum string len :', max([len(s)+1 for s in list_structure]))

def make_arrays(tokens, maxvocab=6000, maxlen=12, pad = '_PAD_', unk = '_UNK_', sos = 'Ⓑ', padfrom = 'end'):
    """integer-index and pad tokenized text"""
    from collections import Counter
    import numpy as np
    
    # get a single list of all words
    words = [word for sent in tokens for word in sent]
    # get the count of each word and sort by frequency (highest to lowest)
    # this is just by convention
    counts = Counter(words)
    words = sorted(list(set(words)), key=counts.get, reverse=True)
    # truncate to desired vocabulary and add PAD and UNK/OOV symbols
    words = words[:maxvocab-3]
    words.insert(0, sos)
    words.insert(0, pad)
    words.insert(-1, unk)
    # create dictionaries
    tok2idx = dict(zip(words, [i for i in range(len(words))]))
    idx2tok = dict(zip([i for i in range(len(words))], words))
    
    # index each sentence
    idxes = []
    for tok_sent in tokens:
        # pad and truncate
        tok_sent = tok_sent[:maxlen-1]
        tok_sent.insert(0, sos)
        while len(tok_sent) < maxlen:
            if padfrom == 'end':
                tok_sent.append(pad)
            else:
                tok_sent.insert(0, pad)
        # convert to indices and add
        idxes.append([tok2idx.get(word, tok2idx[unk]) for word in tok_sent])
    
    # convert to numpy array and return
    return np.array(idxes), tok2idx, idx2tok

# index and pad
seq_idxs, seq2idx, idx2seq = make_arrays(list_sequence, maxvocab=10, maxlen=fuck_var, padfrom = 'start')
str_idxs, str2idx, idx2str = make_arrays(list_structure, maxvocab=10, maxlen=fuck_var, padfrom = 'end')

test_seq_idxs, test_seq2idx, test_idx2seq = make_arrays(list_test_seq, maxvocab=10, maxlen=fuck_var, padfrom = 'start')
test_str_idxs, test_str2idx, test_idx2str = make_arrays(list_test_str, maxvocab=10, maxlen=fuck_var, padfrom = 'end')

# print('seq_idxs = ' + str(seq_idxs))
# print('seq2idx = ' + str(seq2idx))
# print('idx2seq = ' + str(idx2seq))
# print('str_idxs = ' + str(str_idxs))
# print('str2idx = ' + str(str2idx))
# print('idx2str = ' + str(idx2str))

# create output targets from decoder inputs
# this shifts the alignment my one (= predict *next* character) and adds extra dim
str_outs = np.expand_dims(np.hstack((str_idxs[:,1:], np.zeros(shape=(str_idxs.shape[0], 1)))), axis=-1)
test_str_outs = np.expand_dims(np.hstack((test_str_idxs[:,1:], np.zeros(shape=(test_str_idxs.shape[0], 1)))), axis=-1)
# print('str_outs = ' + str(str_outs))

# hyperparameters
SEQ_VOCAB  = len(seq2idx)      # how many unique words in the input language
SEQ_EMBED  = 300               # low long are the character vectors in our input embedding space
MAX_IN_LEN = seq_idxs.shape[1] # how long is the sentence vector

STR_VOCAB    = len(str2idx)    # how many unique words in the output language
STR_EMBED    = 300             # low long are the character vectors in our input embedding space
MAX_OUT_LEN  = str_idxs.shape[1]

# rnn size
HIDDEN_SIZE = 300              # how big is the recurrent cell
DROP_RATE   = 0.4             # what is our dropout frequency

# print('MAX_IN_LEN = ' + str(MAX_IN_LEN))
# print('MAX_OUT_LEN = ' + str(MAX_OUT_LEN))

# bidirectional encoder
encoder_input = Input(shape=(MAX_IN_LEN,), name='encoder_input')

# korean embedding layer and dropout
encoder_embed = Embedding(SEQ_VOCAB, SEQ_EMBED, mask_zero=True, name='encoder_embed')(encoder_input)
encoder_embed = Dropout(DROP_RATE)(encoder_embed)

# two-layer bidirectional LSTM with final states
# divide the HIDDEN_SIZE by two because this is TWO LSTMS stacked
encoder_hout1, fwd_h1, fwd_c1, bck_h1, bck_c1 = Bidirectional(LSTM(int(HIDDEN_SIZE/2), return_sequences=True, return_state=True), name='encoder_lstm1')(encoder_embed)
encoder_hout2, fwd_h2, fwd_c2, bck_h2, bck_c2 = Bidirectional(LSTM(int(HIDDEN_SIZE/2), return_sequences=True, return_state=True), name='encoder_lstm2')(encoder_hout1)

# state concatenation (h, c states for layers 1 and 2)
state_h1 = concatenate([fwd_h1, bck_h1])
state_c1 = concatenate([fwd_c1, bck_c1])
state_h2 = concatenate([fwd_h2, bck_h2])
state_c2 = concatenate([fwd_c2, bck_c2])

# decoder
decoder_input = Input(shape=(MAX_OUT_LEN,), name='decoder_input')

# english embedding layer and dropout
decoder_embed = Embedding(STR_VOCAB, STR_EMBED, mask_zero=True, name='decoder_embed')(decoder_input)
decoder_embed = Dropout(DROP_RATE)(decoder_embed)

# two-layer LSTM initialized with encoder states
decoder_hout1 = LSTM(HIDDEN_SIZE, return_sequences=True, name='decoder_lstm1')(decoder_embed, initial_state=[state_h1, state_c1])
decoder_hout2 = LSTM(HIDDEN_SIZE, return_sequences=True, name='decoder_lstm2')(decoder_hout1, initial_state=[state_h2, state_c2])

# Luong global dot attention
# score function from the Luong apper = dot
score     = dot([decoder_hout2, encoder_hout2], axes=[2, 2], name='attn_dotprod')
# turn score to "attention dist." for weighted sum
attention = Activation('softmax', name='attn_softmax')(score)

# do the attention-weighted sum using dot product
context   = dot([attention, encoder_hout2], axes=[2, 1], name='cont_dotprod')

# 'stacked' the context vector with the decoder guess == 'attention vector'
context   = concatenate([context, decoder_hout2], name='cont_concats')

# activation
context   = TimeDistributed(Dense(HIDDEN_SIZE*2, activation='tanh'), name='cont_dnstanh')(context)

# guess which english letter
output    = TimeDistributed(Dense(STR_VOCAB, activation='softmax'))(context)

# our model takes as input the encoder and decoder, and as target the shifted output we made
model = Model([encoder_input, decoder_input], [output])

model.load_weights('seq2str_{}_epochs.h5'.format(100))
# model.load_weights("weights.best.hdf5")

# compile the model with defined optimizer and loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# model.summary()

EPOCHS = 100

# checkpoint
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# history = model.fit([seq_idxs, str_idxs], str_outs,
#                     batch_size=128,
#                     epochs=EPOCHS,
#                     validation_data=([test_seq_idxs, test_str_idxs], test_str_outs),
#                     callbacks=callbacks_list,
#                     verbose=1)

# save

# this is the model architecture (no weights)
# json_model = model.to_json()
# with open('seq2str_{}_epochs.json'.format(EPOCHS), 'w') as outfile:
#     outfile.write(json_model)

# # this is model architecture (minus some params), and weights
# model.save('seq2str_{}_epochs.h5'.format(EPOCHS))

model = load_model('seq2str_{}_epochs.h5'.format(EPOCHS))
model.load_weights('seq2str_{}_epochs.h5'.format(EPOCHS))

# model.load_weights("weights.best.hdf5")

# attention
heatmapmodel = Model([encoder_input, decoder_input], [output, attention])

def translate(seq_string_list, maxlen = maxlen, maxout = maxlen, sos = 'Ⓑ', unk = '_UNK_', pad = '_PAD_', padfrom = 'start'):
    str_length = list()
    for i, seq_str in enumerate(seq_string_list):
        str_length.append(len(seq_str.split(' ')))
    maxlen = 149
    maxout = maxlen

    seq_tokens = list()
    for i in range(len(seq_string_list)):
        seq_tokens.append([sos] + list(seq_string_list[i].split(' ')))

    seq_toks = list()
    for i, var in enumerate(seq_tokens):
        seq_toks.append(var[:])
    
    for i, var in enumerate(seq_tokens):
        while len(var) <= maxlen:
            if padfrom == 'end':
                var.append(pad)
            else:
                var.insert(0, pad)
   
    encoder_input = np.array([[seq2idx.get(tok, seq2idx[unk]) for tok in i] for i in seq_tokens]) 

    # starting outputs
    decoder_input = np.zeros(shape=(len(seq_string_list), maxlen+1))
    decoder_input[:,0] = str2idx[sos]

    # greedy decoding
    for i in range(0, maxout-1):
        output, heatmap = heatmapmodel.predict_on_batch([encoder_input, decoder_input])
        output = output.argmax(axis=2)
        decoder_input[:,i] = output[:,i]

    print("Batch pred completed")

    def decode(idxes, idx2tok, pad = '_PAD_', unk = '_UNK_', sos='Ⓑ'):
        print("Decoding.... ", idxes)
        toks = []
        for lst in idxes:
            sent = []
            for idx in lst:
                if idx2tok[idx] not in (pad, unk, sos):
                    sent.append(idx2tok[idx])
            toks.append(sent)
        return toks
    
    str_toks = decode(output, idx2str)

    for i, var in enumerate(str_toks):
        # print('output RAW: ', ''.join(var))
        var = var[:str_length[i]]
        print('input :', seq_string_list[i])
        print('output: ', ''.join(var))
        print('')

    # trim heatmap
    # 咕咕咕，到时候再说heatmap的事情，先管预测
    # seqtrim = maxlen - len(seq_toks)
    # strtrim = len(str_toks)
    # heatmap = heatmap[0]
    # heatmap = heatmap[:strtrim, seqtrim+1:]
    
    return str_toks, seq_toks, heatmap

test_str_seq = list()
with open('test_set.txt') as f:
    for line in f.readlines():
        line = line.strip()
        test_str_seq.append(str(line))  

BATCH_TEST = 128
test_str_seq = [test_str_seq[i:i+BATCH_TEST] for i in range(0, len(test_str_seq), BATCH_TEST)]

for i, test_str_list in enumerate(test_str_seq):
    print('predict epoch '+ str(i))
    t1 = time()
    str_toks, k, a = translate(test_str_list)
    print('(%4.1f seconds)\n\n' % (time() - t1))
    with open('test_res.txt', 'a') as file_object:
        for i, e in enumerate(str_toks):
            file_object.write(' '.join(e))
            file_object.write('\n')
