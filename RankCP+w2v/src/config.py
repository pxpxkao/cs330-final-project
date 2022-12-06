import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
TORCH_SEED = 129
DATA_DIR = 'data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'

SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses_eng.pkl'

TRAIN_FILE_PATH = 'clause_keywords.csv'
W2V_FILE = 'w2v_200.txt'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.embedding_dim = 200
        self.embedding_dim_pos = 50
        self.max_sen_len = 30
        self.max_doc_len = 41
        self.keep_prob = 0.8
        self.n_hidden = 200

        self.bert_cache_path = 'bert-base-uncased'
        # self.feat_dim = 768

        self.feat_dim = 400

        # self.gnn_dims = '192'
        self.gnn_dims = '100'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 15
        # self.lr = 0.005 
        # self.batch_size = 32
        self.lr = 1e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

