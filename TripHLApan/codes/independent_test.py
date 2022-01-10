import re
import os
import time
from numpy import *
import pandas as pd

from helper import *



# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
USE_CUDA = torch.cuda.is_available()
random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


def independent_test_distribute(independent_data_file_path, model_save_dir_name, out_file_path):

    MODEL_SAVE_PATH = '../models/' + model_save_dir_name + '/'

    independent_pep_seq_list, independent_allele_seq_list = \
        load_independent_data_from_file(independent_data_file_path)

    files = os.listdir(MODEL_SAVE_PATH)  # trans from 200 epoch
    latest_files = []
    for i in range(5):
        fold_num = i + 1
        matched_file = []
        matched_file_tmp = []
        for file in files:
            matched_file_tmp.append(re.findall('validate_param_fold' + str(fold_num) + r'epoch.*', file))

        for item in matched_file_tmp:
            if item == []:
                continue
            else:
                matched_file.append(item[0])
        if len(matched_file) >= 2:
            mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[0])
            print("mtime:", mtime)
            latest_file_idx = 0
            for idx in range(1, len(matched_file)):
                if os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx]) > mtime:
                    mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx])
                    latest_file_idx = idx
            latest_files.append(MODEL_SAVE_PATH + matched_file[latest_file_idx])
        elif len(matched_file) == 1:
            latest_files.append(MODEL_SAVE_PATH + matched_file[0])
        else:
            continue


    dict_key_prob = {}

    for item in latest_files:
        # MyDataSet type needs changes with model configue changes
        independent_dataset = MyDataSet_distribute(independent_pep_seq_list, independent_allele_seq_list)
        independent_dataloader = tud.DataLoader(independent_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=0)

        model_test = Network_conn()
        #print("CPU")
        if USE_CUDA:
            print('using cuda')
            model_test = model_test.cuda()

        item_name = re.split('[/]', item)[-1]
        fold = int(item_name[len('validate_param_fold')])
        predic_keys, pred_prob = test_independent_only_return_list_triple(model_test, independent_dataloader, fold, item, USE_CUDA, threshold)

        dict_key_prob_tmp = {}
        for idx in range(len(predic_keys)):
            #if predic_keys[idx] in dict_key_prob_tmp.keys():
                #print(real_labels[idx][0], dict_key_real_tmp[predic_keys[idx]])
            dict_key_prob_tmp[predic_keys[idx]] = pred_prob[idx][0]

        if dict_key_prob == {}:
            for item in dict_key_prob_tmp.keys():
                dict_key_prob[item] = []
                dict_key_prob[item].append(dict_key_prob_tmp[item])
        else:
            for item in dict_key_prob.keys():
                dict_key_prob[item].append(dict_key_prob_tmp[item])

    #print("dict_key_real:", len(dict_key_real))
    #print("dict_key_prob:", len(dict_key_prob))

    real_labels_all = []
    pred_prob_all = []
    for item in dict_key_prob.keys():
        probs = dict_key_prob[item]
        pred_prob_mean = mean(probs)

        pred_prob_all.append(pred_prob_mean)
    #print("real_labels_all:",real_labels_all)
    #print("pred_prob_all:", pred_prob_all)


    dataframe = pd.DataFrame({ 'prediction_prob': pred_prob_all})
    dataframe.to_csv(out_file_path, header=0, index=False, sep=',')
        
        




 # configue
from TripHLApan import *
independent_data_file_path = '../for_prediction/'
test_files = os.listdir(independent_data_file_path)
for test_file in test_files:
    if os.path.isdir(independent_data_file_path + test_file):
        continue
    model_save_dir_name = 'TripHLApan' # select models
    out_file_path = '../for_prediction/outputs/' + test_file
    independent_test_distribute(independent_data_file_path + test_file, model_save_dir_name, out_file_path)
# configue

