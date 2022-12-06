import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class E2E_PextC(nn.Module):
    def __init__(self, configs):
        super(E2E_PextC, self).__init__()
        embedding_dim, embedding_dim_pos = configs.embedding_dim, configs.embedding_dim_pos
        sen_len, doc_len = configs.max_sen_len, configs.max_doc_len
        keep_prob1, keep_prob2, keep_prob3 = configs.keep_prob1, configs.keep_prob2, configs.keep_prob3
        n_hidden, n_class = configs.n_hidden, configs.n_class

        self.embedding_dim = embedding_dim; self.embedding_dim_pos = embedding_dim_pos 
        self.sen_len = sen_len; self.doc_len = doc_len
        self.keep_prob1 = keep_prob1; self.keep_prob2 = keep_prob2
        self.n_hidden = n_hidden; self.n_class = n_class

        self.dropout1 = nn.Dropout(p = 1 - keep_prob1)
        self.dropout2 = nn.Dropout(p = 1 - keep_prob2)
        self.dropout3 = nn.Dropout(p = 1 - keep_prob3)
        self.relu = nn.ReLU()
        self.pos_linear = nn.Linear(2*n_hidden, n_class)
        self.cause_linear = nn.Linear(2*n_hidden, n_class)
        self.pair_linear1 = nn.Linear(4*n_hidden + embedding_dim_pos, n_hidden//2)
        self.pair_linear2 = nn.Linear(n_hidden//2, n_class)
        self.word_bilstm = nn.LSTM(embedding_dim, n_hidden, batch_first = True, bidirectional = True)
        self.cause_bilstm = nn.LSTM(2*n_hidden, n_hidden, batch_first = True, bidirectional = True)
        self.pos_bilstm = nn.LSTM(2*n_hidden + n_class, n_hidden, batch_first = True, bidirectional = True)
        self.attention = Attention(n_hidden, sen_len)

    def get_clause_embedding(self, x):
        '''
        input shape: [batch_size, doc_len, sen_len, embedding_dim]
        output shape: [batch_size, doc_len, 2 * n_hidden]
        '''
        x = x.reshape(-1, self.sen_len, self.embedding_dim)
        x = self.dropout1(x)
        # x is of shape (batch_size * max_doc_len, max_sen_len, embedding_dim)
        x, hidden_states = self.word_bilstm(x.float())
        # x is of shape (batch_size * max_doc_len, max_sen_len, 2 * n_hidden)
        s = self.attention(x).reshape(-1, self.doc_len, 2 * self.n_hidden)
        # s is of shape (batch_size, max_doc_len, 2 * n_hidden)
        return s

    def get_emotion_prediction(self, x):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden + n_class]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.pos_bilstm(x.float())
        # x_context is of shape (batch_size, max_doc_len, 2 * n_hidden)
        x = x_context.reshape(-1, 2 * self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_pos = F.softmax(self.pos_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_pos = pred_pos.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context, pred_pos

    def get_cause_prediction(self, x):
        '''
        input shape: [batch_size, doc_len, 2 * n_hidden]
        output(s) shape: [batch_size, doc_len, 2 * n_hidden], [batch_size, doc_len, n_class]
        '''
        x_context, hidden_states = self.cause_bilstm(x.float())
        # x_context is of shape (batch_size, max_doc_len, 2 * n_hidden)
        x = x_context.reshape(-1, 2 * self.n_hidden)
        x = self.dropout2(x)
        # x is of shape (batch_size * max_doc_len, 2 * n_hidden)
        pred_cause = F.softmax(self.cause_linear(x), dim = -1)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        pred_cause = pred_cause.reshape(-1, self.doc_len, self.n_class)
        # pred_pos is of shape (batch_size * max_doc_len, n_class)
        return x_context, pred_cause

    def get_pair_prediction(self, x1, x2, distance):
        '''
        input(s) shape: [batch_size * doc_len, 2 * n_hidden], [batch_size * doc_len, 2 * n_hidden], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos] 
        output shape: [batch_size, doc_len * doc_len, n_class]
        '''        
        x = create_pairs(x1, x2)
        # x is of shape (batch_size, max_doc_len * max_doc_len, 4 * n_hidden)
        x_distance = torch.cat([x, distance.float()], -1)
        # x_distance is of shape (batch_size, max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        x_distance = x_distance.reshape(-1, 4 * self.n_hidden + self.embedding_dim_pos)
        x_distance = self.dropout3(x_distance)
        # x is of shape (batch_size * max_doc_len * max_doc_len, 4 * n_hidden + embedding_dim_pos)
        pred_pair = F.softmax(self.pair_linear2(self.relu(self.pair_linear1(x_distance))), dim = -1)
        # pred_pair is of shape (batch_size * max_doc_len * max_doc_len, n_class)
        pred_pair = pred_pair.reshape(-1, self.doc_len * self.doc_len, self.n_class)
        # pred_pair is of shape (batch_size, max_doc_len * max_doc_len, n_class)
        return pred_pair

    def forward(self, x, distance):
        '''
        input(s) shape: [batch_size, doc_len, sen_len, embedding_dim], 
                        [batch_size, doc_len * doc_len, embedding_dim_pos]
        output(s) shape: [batch_size, doc_len, n_class], [batch_size, doc_len, n_class], 
                         [batch_size, doc_len * doc_len, n_class]
        '''
        s = self.get_clause_embedding(x)
        x_cause, pred_cause = self.get_cause_prediction(s)
        s_pred_cause = torch.cat([s, pred_cause], 2)
        x_pos, pred_pos = self.get_emotion_prediction(s_pred_cause)
        pred_pair = self.get_pair_prediction(x_pos, x_cause, distance)
        return pred_pos, pred_cause, pred_pair

############################################ ATTENTION #######################################################
class Attention(nn.Module):
    def __init__(self, n_hidden, sen_len):
        super(Attention, self).__init__()
        self.n_hidden = n_hidden
        self.sen_len = sen_len
        self.linear1 = nn.Linear(n_hidden*2, n_hidden*2)
        self.linear2 = nn.Linear(n_hidden*2, 1)

    def forward(self, x):
        '''
        input shape: [batch_size * doc_len, sen_len, 2 * n_hidden]
        output shape: [batch_size * doc_len, 2 * n_hidden]
        '''
        x_tmp = x.reshape(-1, self.n_hidden*2)
        # x_tmp is of shape (batch_size * doc_len * sen_len, 2 * n_hidden)
        u = torch.tanh(self.linear1(x_tmp))
        # u is of shape (batch_size * doc_len * sen_len, 2 * n_hidden)
        alpha = self.linear2(u)
        # alpha is of shape (batch_size * doc_len * sen_len, 1)
        alpha = F.softmax(alpha.reshape(-1, 1, self.sen_len), dim = -1)
        # alpha is of shape (batch_size * doc_len, 1, sen_len)
        x = torch.matmul(alpha, x).reshape(-1, self.n_hidden*2)
        # x is of shape (batch_size * doc_len, 2 * n_hidden)
        return x