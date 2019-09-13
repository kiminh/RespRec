import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class VOCAB_processor(object):
    def __init__(self, max_leng):
        self.max_len = max_leng
        self.vocab = {"<PAD>" : 0, "<unk>": 1}
        self.reverse_vocab = {0 : "<PAD>", 1 : "<unk>"}
        self.vocab_size = 2
        self.total_unk = 0
        self.total_token = 0

    def fit(self, x, d = None):
        size = self.vocab_size
        for s in x:
            s = s.split(' ')
            for w in s:
                if d != None and w not in d:
                    self.vocab[w] = 1
                elif self.vocab.get(w, -1) == -1:
                    self.vocab[w] = size
                    self.reverse_vocab[size] = w
                    size += 1
        self.vocab_size = size
        return self.vocab_size

    def transform(self, x):
        self.total_unk = 0
        self.total_token = 0
        trans = []
        for s in x:
            s = s.split(" ")
            array = []
            for ind, w in enumerate(s):
                if ind >= self.max_len:
                    break
                if w in self.vocab:
                    array.append(self.vocab[w])
                else:
                    array.append(self.vocab[ "<unk>" ])
                    self.total_unk += 1
                self.total_token += 1
            array = array + [0] * ( self.max_len - len(s) )
            trans.append( array )
        
        return np.array( trans, dtype = "int32" )



def load_cr(filename):
    lines = list( open(filename, "r").readlines() )
    lines = [ l.strip() for l in lines ]
    
    sents = []
    labels = []
    domains = []
    for l in lines:
       
        label, sent, domain = l.split('\t')

        # sents.append( clean_str(sent) )
        sents.append( sent )
        labels.append( int(label) )
        domains.append( int(domain) )
    return sents, labels, domains


def load_data():
    sents, labels, domains = load_cr(filename = "./BDEKa.lab.data" )
    assert( len(sents) == len(labels) and len(sents) == len(domains) )
    print("Totally load {} data".format( len(sents) ))

    unsup_sents, unsup_labels, unsup_domains = load_cr( filename = "./BDEKa.unl.data" )
    assert( len(unsup_sents) == len(unsup_labels) and len(unsup_sents) == len(unsup_domains) )
    print("Totally load {} data".format( len(unsup_sents) ))

    max_sent_length = max( [len(x.split(' ')) for x in sents] )
    print("Max sentence Length: {} ".format(max_sent_length) )
    if max_sent_length > 256:
        max_sent_length = 256
        print("Max sentence Length trimmed to {} ".format(max_sent_length) )

    # Vocab2index ####
    vocab_processor = VOCAB_processor( max_sent_length )
    vocab_size = vocab_processor.fit( sents )
    print("Vocabulary Size: {:d}".format( vocab_size ))
    x = vocab_processor.transform( sents )

    unsup_x = vocab_processor.transform( unsup_sents )

    # labels 2 onehot ####
    num_label = 2
    y = np.zeros( (len(sents), num_label ) )
    y[ np.arange(len(sents)), np.array(labels) ] = 1

    # no labels
    # unsup_y = np.zeros( (len(unsup_sents), num_label ) )
    # unsup_y[ np.arrange(len(unsup_sents)), np.array(unsup_labels)] = 1

    # num_domains = max(domains) ###
    num_domain = 13
    #num_domain = max(domains)
    # d = np.zeros( (len(sents), num_domain ) )
    # d[ np.arange(len(sents)), np.array(domains) ] = 1
    d = np.array( domains ).reshape( (len(sents), 1) )

    # unsup_d = np.zeros( (len(unsup_sents), num_domain ) )
    # unsup_d[ np.arange( len(unsup_sents)), np.array(unsup_domains) ] = 1
    unsup_d = np.array( unsup_domains ).reshape( (len(unsup_sents), 1) )

    return max_sent_length, vocab_size, num_label, num_domain, \
        x, y, d, unsup_x, unsup_d



class batch_iter(object):
    #data := list of np.darray
    def __init__(self, data, batch_size, is_shuffle=True):
        assert( len(data) > 0 )
        self.data = data
        self.batch_size = batch_size
        self.data_size = len( data[0] )
        assert (self.data_size >= self.batch_size)

        self.index = self.data_size
        self.is_shuffle = is_shuffle

    def fetch_batch(self, start, end):
        batch_list = []
        for data in self.data:
            batch_list.append(data[start: end])
        return batch_list

    def shuffle(self):
        shuffle_indices = np.random.permutation( np.arange(self.data_size) )
        for i in range(len(self.data)):
            self.data[i] = (self.data[i])[shuffle_indices]

    def next_full_batch(self):
        if self.index < self.data_size - self.batch_size:
            self.index += self.batch_size
            return self.fetch_batch(self.index - self.batch_size, self.index)
        else:
            if self.is_shuffle:
                self.shuffle()
            self.index = self.batch_size
            return self.fetch_batch(0, self.batch_size)


#this is a quick iter not for general usage
class cross_validation_iter(object):

    def __init__(self, data, fold = 10):
        for i in [1, 2]:
            assert( len(data[0]) == len(data[i]) )
        self.x = data[0]
        self.y = data[1]
        self.d = data[2]
        # self.x2 = data[3]
        # self.d2 = data[4]
        self.fold = self.d.shape[1]
        self.cv = [0, 1, 2, 3]

    def fetch_next(self):
        x_train = [ ]
        y_train = [ ]
        d_train = [ ]
        # x2_train = [ ]
        # d2_train = [ ]

        x_test = [ ]
        y_test = [ ]
        d_test = [ ]

        cv = self.cv
        for i in range( len(self.x) ):
            if self.d[i] in self.cv:
                x_test.append(self.x[i])
                y_test.append(self.y[i])
                d_test.append(self.d[i])
        
            else:
                x_train.append(self.x[i])
                y_train.append(self.y[i])
                d_train.append(self.d[i])
                # x2_train.append( self.x2[i] )
                # d2_train.append( self.d2[i] )

        for i in range( len(self.cv) ):
            self.cv[i] = (self.cv[i] + 4) % self.fold
        
        return np.array(x_train), np.array(y_train), np.array(d_train), \
            np.array(x_test), np.array(y_test), np.array(d_test)
            # np.array(x2_train), np.array(d2_train), \

# class cross_validation_indomain_iter(object):

#     def __init__(self, data, fold = 10):
#         for i in range(1, len(data)):
#             assert( len(data[0]) == len(data[i]) )
#         self.x = data[0]
#         self.y = data[1]
#         self.d = data[2]
#         self.fold = 10
#         self.cv = 0

#     def fetch_next(self):
#         x_train = [ ]
#         y_train = [ ]
#         d_train = [ ]

#         x_test = [ ]
#         y_test = [ ]
#         d_test = [ ]

#         cv = self.cv
#         cv_domain = []
#         for _ in range( (self.d).shape[1] ):
#             cv_domain.append( 0 )
#             x_test.append( [ ] )
#             y_test.append( [ ] )
#             d_test.append( [ ] )
        
#         for i in range( len(self.x) ):
#             dom = np.argmax( self.d[i] )
#             if cv_domain[dom] % self.fold == cv:
#                 x_test[dom].append(self.x[i])
#                 y_test[dom].append(self.y[i])
#                 d_test[dom].append(self.d[i])
#             else:
#                 x_train.append(self.x[i])
#                 y_train.append(self.y[i])
#                 d_train.append(self.d[i])
#             cv_domain[dom] += 1

#         self.cv = (self.cv + 1) % self.fold
        
#         return np.array(x_train), np.array(y_train), np.array(d_train), \
#             np.array(x_test), np.array(y_test), np.array(d_test)

def data_split(x, y, d, ratio = 0.5):
    assert( len(x) == len(y) )
    assert( len(x) == len(d) )

    data_size = len(x)
    shuffle_indices = np.random.permutation( np.arange( data_size ) )
    x = x[ shuffle_indices ]
    y = y[ shuffle_indices ]
    d = d[ shuffle_indices ]

    A_len = int(data_size * ratio)
    C_len = data_size - A_len
    num_domain = 13
    # d_unk = np.zeros( (C_len, num_domain) )
    d_unk = np.full( (C_len, 1), fill_value=num_domain )

    return x[:A_len, :], y[:A_len, :], d[:A_len, :], x[A_len:, :], y[A_len:, :], d_unk

def indomain_data_split(x, y, d, ratio = 0.5, ind=0):
    assert( len(x) == len(y) )
    assert( len(x) == len(d) )

    data_size = len(x)
    shuffle_indices = np.random.permutation( np.arange( data_size ) )
    x = x[ shuffle_indices ]
    y = y[ shuffle_indices ]
    d = d[ shuffle_indices ]

    A_len = int(data_size * ratio)
    C_len = data_size - A_len
    num_domain = 13
    # d_unk = np.zeros( (C_len, num_domain) )
    d_unk = np.full( (C_len, 1), fill_value=num_domain )
    tx, ty, td = x[A_len:, :], y[A_len:, :], d[A_len:, :]
    rx ,ry, rd = [], [], []
    for i in range( len(tx) ):
        if td[i][0] == ind:
            rx.append( tx[i] )
            ry.append( ty[i] )
            rd.append( td[i] )
    rx = np.array( rx )
    ry = np.array( ry )
    rd = np.array( rd )
    # print(rx.shape)
    # print(ry.shape)
    # print(rd.shape)
    # exit()

    return x[:A_len, :], y[:A_len, :], d[:A_len, :], rx, ry, rd

def split_dev_test( x, y, d, ratio = 0.1 ):
    assert( len(x) == len(y) )
    assert( len(x) == len(d) )

    data_size = len(x)
    shuffle_indices = np.random.permutation( np.arange( data_size ) )
    x = np.array( x )[ shuffle_indices ]
    y = np.array( y )[ shuffle_indices ]
    d = np.array( d )[ shuffle_indices ]

    dev_len = int(data_size * ratio)
    return x[:dev_len, :], y[:dev_len, :], d[:dev_len, :], x[dev_len:, :], y[dev_len:, :], d[dev_len:, :]
