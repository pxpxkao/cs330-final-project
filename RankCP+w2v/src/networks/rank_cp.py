from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, DATA_DIR
from transformers import BertModel
from networks.gnn_layer import GraphAttentionLayer


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.emb = ClauseEmbedding(configs)
        self.gnn = GraphNN(configs)
        self.pred = Pre_Predictions(configs)
        self.rank = RankNN(configs)
        self.pairwise_loss = configs.pairwise_loss

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, adj, x_b):
        # bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
        #                         attention_mask=bert_masks_b.to(DEVICE),
        #                         token_type_ids=bert_segment_b.to(DEVICE))
        # doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))

        doc_sents_h = self.emb(x_b)
        doc_sents_h = doc_sents_h[:, :max(doc_len), :]

        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)
        pred_e, pred_c = self.pred(doc_sents_h)

        couples_pred, emo_cau_pos = self.rank(doc_sents_h)

        return couples_pred, emo_cau_pos, pred_e, pred_c

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h

    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        if not self.pairwise_loss:
            couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
            couples_true = torch.FloatTensor(couples_true).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            couples_true = couples_true.masked_select(couples_mask)
            couples_pred = couples_pred.masked_select(couples_mask)
            loss_couple = criterion(couples_pred, couples_true)
        else:
            x1, x2, y = self.pairwise_util(couples_pred, couples_true, couples_mask)
            criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            loss_couple = criterion(F.tanh(x1), F.tanh(x2), y)

        return loss_couple, doc_couples_pred

    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=min(3, couples_pred_i.size(0)), dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        pred_e = pred_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)

        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)
        return loss_e, loss_c

    def pairwise_util(self, couples_pred, couples_true, couples_mask):
        """
        TODO: efficient re-implementation; combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()
        x1, x2 = [], []
        for i in range(batch):
            x1_i_tmp = []
            x2_i_tmp = []
            couples_mask_i = couples_mask[i]
            couples_pred_i = couples_pred[i]
            couples_true_i = couples_true[i]
            for pred_ij, true_ij, mask_ij in zip(couples_pred_i, couples_true_i, couples_mask_i):
                if mask_ij == 1:
                    if true_ij == 1:
                        x1_i_tmp.append(pred_ij.reshape(-1, 1))
                    else:
                        x2_i_tmp.append(pred_ij.reshape(-1))
            m = len(x2_i_tmp)
            n = len(x1_i_tmp)
            x1_i = torch.cat([torch.cat(x1_i_tmp, dim=0)] * m, dim=1).reshape(-1)
            x1.append(x1_i)
            x2_i = []
            for _ in range(n):
                x2_i.extend(x2_i_tmp)
            x2_i = torch.cat(x2_i, dim=0)
            x2.append(x2_i)

        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        y = torch.FloatTensor([1] * x1.size(0)).to(DEVICE)
        return x1, x2, y

class ClauseEmbedding(nn.Module):
    def __init__(self, configs):
        super(ClauseEmbedding, self).__init__()
        self.sen_len = configs.max_sen_len; self.doc_len = configs.max_doc_len
        self.embedding_dim = configs.embedding_dim; self.n_hidden = configs.n_hidden

        self.dropout = nn.Dropout(p = 1 - configs.keep_prob)
        self.word_bilstm = nn.LSTM(self.embedding_dim, self.n_hidden, batch_first = True, bidirectional = True)
        self.attention = Attention(self.n_hidden, self.sen_len)

        self.word_embedding = torch.load(join(DATA_DIR, configs.split, 'word_embedding.pth'))
    
    def forward(self, x):
        '''
        input shape: [batch_size, doc_len, sen_len, embedding_dim]
        output shape: [batch_size, doc_len, 2 * n_hidden]
        '''
        x = F.embedding(torch.from_numpy(x).type(torch.LongTensor), self.word_embedding).to(DEVICE)
        # x = F.embedding(x.type(torch.LongTensor), self.word_embedding).to(DEVICE)
        x = x.reshape(-1, self.sen_len, self.embedding_dim)
        x = self.dropout(x)
        # x is of shape (batch_size * max_doc_len, max_sen_len, embedding_dim)
        x, hidden_states = self.word_bilstm(x.float())
        # x is of shape (batch_size * max_doc_len, max_sen_len, 2 * n_hidden)
        s = self.attention(x).reshape(-1, self.doc_len, 2 * self.n_hidden)
        # s is of shape (batch_size, max_doc_len, 2 * n_hidden)
        return s


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

class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)

        return doc_sents_h


class RankNN(nn.Module):
    def __init__(self, configs):
        super(RankNN, self).__init__()
        self.K = configs.K
        self.pos_emb_dim = configs.pos_emb_dim
        self.pos_layer = nn.Embedding(2*self.K + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.rank_feat_dim = 2*self.feat_dim + self.pos_emb_dim
        self.rank_layer1 = nn.Linear(self.rank_feat_dim, self.rank_feat_dim)
        self.rank_layer2 = nn.Linear(self.rank_feat_dim, 1)

    def forward(self, doc_sents_h):
        batch, _, _ = doc_sents_h.size()
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        couples = torch.cat([couples, rel_pos_emb], dim=2)

        couples = F.relu(self.rank_layer1(couples))
        couples_pred = self.rank_layer2(couples)
        return couples_pred.squeeze(2), emo_cau_pos

    def couple_generator(self, H, k):
        batch, seq_len, feat_dim = H.size()
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)

            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)
        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))


class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
