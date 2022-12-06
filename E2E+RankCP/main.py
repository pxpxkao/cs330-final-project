import sys, os, time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from models import *
from config import *
from data_loader import *
from transformers import AdamW, get_linear_schedule_with_warmup

def train_and_eval(configs, Model, pos_cause_criterion, optimizer):
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(
        configs.embedding_dim, configs.embedding_dim_pos, configs.train_file_path, configs.w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    torch.save(word_embedding, './save/word_embedding.pth')
    torch.save(word_id_mapping, './save/word_id_mapping.pth')
    acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
    acc_pair_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], []
    #################################### LOOP OVER FOLDS ####################################
    for fold in range(1, 11):
        print('############# fold {} begin ###############'.format(fold))
        ############################# RE-INITIALIZE MODEL PARAMETERS #############################
        for layer in Model.parameters():
            nn.init.uniform_(layer.data, -0.10, 0.10)
        #################################### TRAIN/TEST DATA ####################################
        train_file_name = 'fold{}_train.txt'.format(fold)
        val_file_name = 'fold{}_val.txt'.format(fold)
        print("Prepare Training Data.........")
        tr_y_position, tr_y_cause, tr_y_pair, tr_x, tr_sen_len, tr_doc_len, tr_distance, \
            tr_y_mask, tr_y_couple, tr_doc_ids = load_data_pair(
                        DATA_DIR+train_file_name, word_id_mapping, configs.max_doc_len, configs.max_sen_len)
        val_y_position, val_y_cause, val_y_pair, val_x, val_sen_len, val_doc_len, val_distance, \
            val_y_mask, val_y_couple, val_doc_ids = load_data_pair(
                DATA_DIR+val_file_name, word_id_mapping, configs.max_doc_len, configs.max_sen_len)
        max_f1_cause, max_f1_pos, max_f1_pair, max_f1_avg = [-1.] * 4

        num_steps_all = len(tr_y_position) // configs.gradient_accumulation_steps * configs.epochs
        warmup_steps = int(num_steps_all * configs.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
        #################################### LOOP OVER EPOCHS ####################################
        for epoch in range(1, configs.epochs + 1):
            start_time = time.time()
            step = 1
            #################################### GET BATCH DATA ####################################
            for train, _ in get_batch_data_pair(
                tr_x, tr_sen_len, tr_doc_len, tr_y_position, tr_y_cause, tr_y_pair, tr_distance, tr_y_mask, tr_y_couple, tr_doc_ids, configs.batch_size):

                tr_x_batch, tr_sen_len_batch, tr_doc_len_batch, tr_true_y_pos, tr_true_y_cause, \
                tr_true_y_pair, tr_distance_batch, tr_y_mask_batch, tr_y_couple_batch, tr_doc_ids_batch = train
                Model.train()
                tr_input = embedding_lookup(word_embedding, tr_x_batch)
                # tr_pred_y_pos, tr_pred_y_cause, tr_pred_y_pair = Model(tr_input)
                tr_couples_pred, tr_emo_cau_pos, tr_pred_y_pos, tr_pred_y_cause = Model(tr_input)
                ############################## LOSS FUNCTION AND OPTIMIZATION ##############################
                loss_e = pos_cause_criterion(tr_true_y_pos, tr_pred_y_pos, tr_doc_len_batch)
                loss_c = pos_cause_criterion(tr_true_y_cause, tr_pred_y_cause, tr_doc_len_batch)
                loss_couple, doc_couples_pred = Model.rank_cp.loss_rank(tr_couples_pred, tr_emo_cau_pos, tr_y_couple_batch, tr_y_mask_batch, test=True)
                loss = loss_e*configs.pos + loss_c*configs.cause + loss_couple.to('cpu')*configs.pair

                loss = loss / configs.gradient_accumulation_steps

                loss.backward()
                if step % configs.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    Model.zero_grad()
                #################################### PRINT AFTER EPOCHS ####################################
                if step % 100 == 0:
                    print('Fold {}, Epoch {}, step {}: train loss {:.4f} '.format(fold, epoch, step, loss))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_pos, tr_true_y_pos, tr_doc_len_batch)
                    print('emotion_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))
                    acc, p, r, f1 = acc_prf_aux(tr_pred_y_cause, tr_true_y_cause, tr_doc_len_batch)
                    print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            acc, p, r, f1))

                    doc_couples_pred_all = lexicon_based_extraction(tr_doc_ids_batch, doc_couples_pred)
                    metric_ec, _, _ = eval_func(tr_y_couple_batch, doc_couples_pred_all)
                    p, r, f1 = metric_ec[0], metric_ec[1], metric_ec[2]
                    print('pair_predict: train p {:.4f} r {:.4f} f1 score {:.4f}'.format(
                            p, r, f1))
                step += 1
            #################################### TEST ON 1 FOLD ####################################
            with torch.no_grad():
                Model.eval()
                val_input = embedding_lookup(word_embedding, val_x)
                val_couples_pred, val_emo_cau_pos, val_pred_y_pos, val_pred_y_cause = Model(val_input)

                loss_e = pos_cause_criterion(val_y_position, val_pred_y_pos, val_doc_len)
                loss_c = pos_cause_criterion(val_y_cause, val_pred_y_cause, val_doc_len)
                loss_couple, doc_couples_pred = Model.rank_cp.loss_rank(val_couples_pred, val_emo_cau_pos, val_y_couple, val_y_mask, test=True)
                loss = loss_e*configs.pos + loss_c*configs.cause + loss_couple.to('cpu')*configs.pair
                print('Fold {} val loss {:.4f}'.format(fold, loss))

                acc, p, r, f1 = acc_prf_aux(val_pred_y_pos, val_y_position, val_doc_len)
                result_avg_pos = [acc, p, r, f1]
                if f1 > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc, p, r, f1
                print('emotion_predict: val acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos))

                acc, p, r, f1 = acc_prf_aux(val_pred_y_cause, val_y_cause, val_doc_len)
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('cause_predict: val acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause))

                doc_couples_pred_all = lexicon_based_extraction(val_doc_ids, doc_couples_pred)
                metric_ec, _, _ = eval_func(val_y_couple, doc_couples_pred_all)
                p, r, f1 = metric_ec[0], metric_ec[1], metric_ec[2]
                result_avg_pair = [p, r, f1]
                if f1 > max_f1_pair:
                    max_p_pair, max_r_pair, max_f1_pair = p, r, f1
                print('pair_predict: val p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    max_p_pair, max_r_pair, max_f1_pair))
            
            #################################### STORE BETTER PAIR F1 ####################################
            if result_avg_pair[-1] > max_f1_avg:
                # torch.save(pos_embedding, "./save/pos_embedding_fold_{}.pth".format(fold))
                torch.save(Model.state_dict(), "./save/E2E-PextE_RankCP_fold_{}.pth".format(fold))
                # torch.save(Model.state_dict(), "./save/E2E-PextC_RankCP_fold_{}.pth".format(fold))
                max_f1_avg = result_avg_pair[-1]
                result_avg_cause_max = result_avg_cause
                result_avg_pos_max = result_avg_pos
                result_avg_pair_max = result_avg_pair

            print('avg max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2], result_avg_cause_max[3]))
            print('avg max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                result_avg_pos_max[0], result_avg_pos_max[1], result_avg_pos_max[2], result_avg_pos_max[3]))
            print('avg max pair: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                result_avg_pair_max[0], result_avg_pair_max[1], result_avg_pair_max[2]))
            print("\nEpoch {}... Total time {} mins".format(epoch, (time.time()-start_time)/60))

        print('############# fold {} end ###############'.format(fold))
        acc_cause_list.append(result_avg_cause_max[0])
        p_cause_list.append(result_avg_cause_max[1])
        r_cause_list.append(result_avg_cause_max[2])
        f1_cause_list.append(result_avg_cause_max[3])
        acc_pos_list.append(result_avg_pos_max[0])
        p_pos_list.append(result_avg_pos_max[1])
        r_pos_list.append(result_avg_pos_max[2])
        f1_pos_list.append(result_avg_pos_max[3])
        p_pair_list.append(result_avg_pair_max[0])
        r_pair_list.append(result_avg_pair_max[1])
        f1_pair_list.append(result_avg_pair_max[2])

    #################################### FINAL TEST RESULTS ON 10 FOLDS ####################################
    all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, \
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list, p_pair_list, r_pair_list, f1_pair_list,]
    acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos, p_pair, r_pair, f1_pair = \
        map(lambda x: np.array(x).mean(), all_results)
    print('\ncause_predict: val f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
    print('emotion_predict: val f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1,1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
    print('pair_predict: val f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
    print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))

def main():
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    configs = Config()
    Model = Network(configs)
    Model.to(DEVICE)
    # print(Model)
    print('Total params:', sum([param.nelement() for param in Model.parameters()]))
    x = torch.rand([configs.batch_size, configs.max_doc_len, configs.max_sen_len, configs.embedding_dim]).to(DEVICE)
    couples_pred, emo_cau_pos, pred_pos, pred_cause = Model(x)
    print("Random i/o shapes x: {}, y_pos: {}, y_cause: {}, couples_pred: {}, emo_cau_pos: {}".format(
        x.shape, pred_pos.shape, pred_cause.shape, couples_pred.shape, np.array(emo_cau_pos).shape))
    pos_cause_criterion = ce_loss_aux()
    optimizer = optim.Adam(Model.parameters(), lr=configs.learning_rate, weight_decay=configs.l2_reg)
    train_and_eval(configs, Model, pos_cause_criterion, optimizer)

if __name__ == '__main__':
    main()