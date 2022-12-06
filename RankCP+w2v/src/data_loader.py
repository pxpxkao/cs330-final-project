import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils.utils import *


torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split
        self.max_doc_len = configs.max_doc_len; self.max_sen_len = configs.max_sen_len

        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, self.x_list = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        x_idx = self.x_list[idx]

        if bert_token_lens > 512:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens,
                                                                          doc_couples, y_emotions, y_causes, doc_len)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        # x_idx = torch.LongTensor(x_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens, x_idx

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        train_file_path = join(self.data_dir, self.split, TRAIN_FILE_PATH)
        w2v_file = join(self.data_dir, self.split, W2V_FILE)
        word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(
            200, 50, train_file_path, w2v_file)
        word_embedding = torch.from_numpy(word_embedding)
        torch.save(word_embedding, join(self.data_dir, self.split, 'word_embedding.pth'))
        torch.save(word_id_mapping, join(self.data_dir, self.split, 'word_id_mapping.pth'))
        word_idx = word_id_mapping

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        x_list = []

        data_list = read_json(data_file)
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)

            y_emotions, y_causes = [], []
            doc_clauses = doc['clauses']
            doc_str = ''

            # x = np.zeros((configs.max_doc_len, configs.max_sen_len))
            x_input = [[0 for _ in range(self.max_sen_len)] for _ in range(self.max_doc_len)]
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)

                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                doc_str += '[CLS] ' + clause['clause'].lower() + ' [SEP] '

                # words = clause['clause']
                # for j, word in enumerate(words.split()):
                #     word = word.lower()
                #     if j >= self.max_sen_len:
                #         break
                #     elif word not in word_idx:
                #         x_input[i][j] = 0 # 24166
                #     else: x_input[i][j] = int(word_idx[word])

            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)

            x_list.append(x_input)

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list, x_list

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_len = i
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b, x_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b, _ = pad_docs(doc_len_b, y_emotions_b, y_causes_b, x_b)
    adj_b = pad_matrices(doc_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.bool)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    return np.array(doc_len_b), np.array(adj_b), \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
           bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, np.array(x_b)


def pad_docs(doc_len_b, y_emotions_b, y_causes_b, x_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    x_b_ = []
    for y_emotions, y_causes, x in zip(y_emotions_b, y_causes_b, x_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_mask = list(map(lambda x: False if x == -1 else True, y_emotions_))
        x_ = np.array(x)[:max_doc_len, :]

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)
        x_b_.append(x_)

    return y_mask_b, y_emotions_b_, y_causes_b_, x_b_


def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad


############################################ LOAD W2V EMBEDDING ###############################################
def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend(emotion.lower().split() + clause.lower().split())
        # words extended by ['happy','the','thief','was','caught']

    words = set(words) # Collection of all unique words
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # Each word and its position
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # Each word and its position

    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) 
            # Randomly take from the uniform distribution [-0.1, 0.1]
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))
    # add a noisy embedding in the end for out of vocabulary words
    embedding.extend([list(np.random.rand(embedding_dim) / 5. - 0.1)])

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) \
        for i in range(200)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


if __name__ == '__main__':
    configs = Config()
    from networks.rank_cp import *
    model = Network(configs).to(DEVICE)
    # print(model)
    test_loader = build_inference_data(configs, fold_id=1, data_type='test')
    for train_step, batch in enumerate(test_loader, 1):
        doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, x_b = batch

        couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b, adj_b, x_b)

        # for var in ['doc_len_b', 'adj_b', 'y_emotions_b', 'y_causes_b', 'y_mask_b', 'bert_token_b', 'bert_segment_b', 'bert_masks_b', 'bert_clause_b']:
        #     print('{}.shape {}'.format( var, eval(var).shape ))
        # print('doc_couples_b:', doc_couples_b)
        # print('doc_id_b:', doc_id_b)
        # print('bert_clause_b:', bert_clause_b)
        # print('bert_token_b', bert_token_b)
        # print('doc_len_b:', doc_len_b)

        break