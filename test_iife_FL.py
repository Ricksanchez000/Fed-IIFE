def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import sys
file_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, str(file_path))
from iife import iife, blockPrint, enablePrint
from data import *
from helper import *
import numpy as np
import json
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, mean_squared_error, r2_score, make_scorer, mean_absolute_error, log_loss
from sklearn.model_selection import KFold
import time
import ray
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from knncmi import *
import pickle
from sklearn.preprocessing import OneHotEncoder
from hyperparam_tune import hyperparam_tune
import logging
import lightgbm as lgb
from time import sleep
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



def _prep_inputs_for_model(model, X_train, X_test, vartype_list):
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    import numpy as np
    if model in ["LR", "lasso"]:
        if "cat" in vartype_list:
            enc = OneHotEncoder(handle_unknown='ignore')
            X_train_num = X_train[:, np.where(np.array(vartype_list) == "num")[0]]
            X_train_cat = X_train[:, np.where(np.array(vartype_list) == "cat")[0]]
            X_test_num  = X_test [:, np.where(np.array(vartype_list) == "num")[0]]
            X_test_cat  = X_test [:, np.where(np.array(vartype_list) == "cat")[0]]
            enc.fit(X_train_cat)
            X_train_ohe = enc.transform(X_train_cat).toarray()
            X_test_ohe  = enc.transform(X_test_cat).toarray()
            X_train_tmp = np.concatenate((X_train_num, X_train_ohe), axis=1)
            X_test_tmp  = np.concatenate((X_test_num , X_test_ohe ), axis=1)
        else:
            X_train_tmp, X_test_tmp = X_train, X_test
        scaler = MinMaxScaler()
        scaler.fit(X_train_tmp)
        return scaler.transform(X_train_tmp), scaler.transform(X_test_tmp)
    else:
        return X_train, X_test


### NEW: 统一创建/更新模型
def _make_clf(model, hyperparams):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if model == "LR":
        return LogisticRegression(C=hyperparams["C"], max_iter=50000)
    if model == "lasso":
        return Lasso(alpha=hyperparams["alpha"], max_iter=500000)
    if model == "RF":
        return RandomForestClassifier(max_depth=hyperparams["max_depth"],
                                      max_features=hyperparams["max_features"],
                                      max_samples=hyperparams["max_samples"],
                                      n_estimators=hyperparams["n_estimators"],
                                      random_state=0)
    if model == "RFR":
        return RandomForestRegressor(max_depth=hyperparams["max_depth"],
                                     max_features=hyperparams["max_features"],
                                     max_samples=hyperparams["max_samples"],
                                     n_estimators=hyperparams["n_estimators"],
                                     random_state=0)
    if model == "lgbm_reg":
        return lgb.LGBMRegressor(n_estimators=hyperparams["n_estimators"],
                                 learning_rate=hyperparams["learning_rate"],
                                 subsample=hyperparams["subsample"],
                                 colsample_bytree=hyperparams["colsample_bytree"],
                                 reg_lambda=hyperparams["reg_lambda"],
                                 random_state=0, num_threads=1)
    if model == "lgbm_class":
        return lgb.LGBMClassifier(n_estimators=hyperparams["n_estimators"],
                                  learning_rate=hyperparams["learning_rate"],
                                  subsample=hyperparams["subsample"],
                                  colsample_bytree=hyperparams["colsample_bytree"],
                                  reg_lambda=hyperparams["reg_lambda"],
                                  random_state=0, num_threads=1)
    raise ValueError(model)



### NEW: 子集加载器；优先调用你已有的 process_openml586_1~4；否则自动做 4 折划分
def load_openml586_client_subset(seed, client_id):
    # 尝试直接使用你可能已经实现的 loader
    try:
        fn_map = {
            1: process_openml586_1,
            2: process_openml586_2,
            3: process_openml586_3,
            4: process_openml586_4,
        }
        return fn_map[client_id](seed)
    except Exception:
        # 回退：从全量数据中按标签分层切成4份，取其中一份当训练，其余当测试（与全局保持分布一致）
        from sklearn.model_selection import StratifiedKFold
        Xtr, ytr, Xte, yte, feat_list, Xorig, vtypes = process_openml586(seed)
        ytr = np.array(ytr)
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        # 找到 client_id 对应的 fold 作为“本客户端训练集”；测试集仍用原全局 Xte/yte（也可用剩余3折拼起来）
        for idx, (tr_idx, _) in enumerate(kf.split(Xtr, ytr), start=1):
            if idx == client_id:
                return (Xtr[tr_idx], ytr[tr_idx], Xte, yte, feat_list, Xorig, vtypes)
        raise RuntimeError("client split failed")



runs = [
#("openml586","lasso")
("openml586","RFR")
]
output_dir = "outdir/iife_outputs/"
save_Xs=True
loss_before=[]
loss_after=[]
scores_before=[]
scores_after=[]
hyperparams_params=[]
hyperparams_params_after=[]
hyperparams_scores=[]
times=[]
#set some ray settings to work properly
runtime_env = {"working_dir": "src"}
ray.init(runtime_env=runtime_env, num_cpus=6)
# run on 4 total runs for this test run
num_seeds=2
num_seed2s=2
for it,r in enumerate(runs):
    for seed in range(0,num_seeds):
        task,model=r
        #####
        ##### load in data
        #####
        if task == "openml586":
            X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_openml586(seed)
        elif task == "cal_housing":
            X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_cal_housing(seed)
        elif task == "jungle_chess":
            X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_jungle_chess(seed)
        train_size = X_train.shape[0]
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        #####
        ##### Normalize and one-hot encode data if linear model
        #####
        if model == "LR" or model == "lasso":
            if "cat" in vartype_list:
                # one hot encode
                enc = OneHotEncoder(handle_unknown='ignore')
                X_train_num = X_train[:,np.where(np.array(vartype_list)=="num")[0]]
                X_train_cat = X_train[:,np.where(np.array(vartype_list)=="cat")[0]]
                X_test_num = X_test[:,np.where(np.array(vartype_list)=="num")[0]]
                X_test_cat = X_test[:,np.where(np.array(vartype_list)=="cat")[0]]
                enc.fit(X_train_cat)
                X_train_ohe = enc.transform(X_train_cat).toarray()
                X_test_ohe = enc.transform(X_test_cat).toarray()

                X_train_temp = np.concatenate((X_train_num,X_train_ohe),axis=1)
                X_test_temp = np.concatenate((X_test_num,X_test_ohe),axis=1)
            else:
                X_train_temp=X_train
                X_test_temp=X_test

            scaler = MinMaxScaler()
            scaler.fit(X_train_temp)
            X_train_temp=scaler.transform(X_train_temp)
            X_test_temp=scaler.transform(X_test_temp)
        else:
            X_train_temp=X_train
            X_test_temp=X_test
        
        #####
        ##### Hyperparameter tune before autofe
        #####
        print("Starting hyperparam tuning")
        hyperparams,hyperscores = hyperparam_tune(task,model,seed)
        print("Ending hyperparam tuning")
        hyperparams_params.append(hyperparams)
        hyperparams_scores.append(hyperscores)

        if model == "LR":
            clf = LogisticRegression(C=hyperparams["C"],max_iter=50000)
        elif model == "RFR":
            clf = RandomForestRegressor(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
        elif model == "RF":
            clf = RandomForestClassifier(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
        elif model == "lasso":
            clf = Lasso(alpha=hyperparams["alpha"],max_iter=50000)
        elif model == "lgbm_reg":
            clf = lgb.LGBMRegressor(
                n_estimators = hyperparams["n_estimators"],
                learning_rate = hyperparams["learning_rate"],
                subsample = hyperparams["subsample"],
                colsample_bytree = hyperparams["colsample_bytree"],
                reg_lambda = hyperparams["reg_lambda"], 
                random_state=0, num_threads=1
            )
        elif model == "lgbm_class":
            clf = lgb.LGBMClassifier(
                n_estimators = hyperparams["n_estimators"],
                learning_rate = hyperparams["learning_rate"],
                subsample = hyperparams["subsample"],
                colsample_bytree = hyperparams["colsample_bytree"],
                reg_lambda = hyperparams["reg_lambda"], 
                random_state=0, num_threads=1
            )

        #####
        ##### Find test scores before AutoFE
        #####

        clf.fit(X_train_temp, y_train)
        pred = clf.predict(X_test_temp)

        if model in ["lasso","RFR","lgbm_reg"]:
            score = RAE_comp(y_test,pred)
        else:
            score = f1_score(y_test,pred,average="micro")


        scores_before.append(score)

        if model=="LR":
            loss_before.append(log_loss(y_train,clf.predict_proba(X_train_temp)) )

        for seed2 in range(0,num_seed2s):
            if task == "openml586":
                X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_openml586(seed)
            elif task == "cal_housing":
                X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_cal_housing(seed)
            elif task == "jungle_chess":
                X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list = process_jungle_chess(seed)
            train_size = X_train.shape[0]
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            data_input = (X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list)
            # blockPrint()
            start1=time.time()
            if model=="lasso":
                clf.set_params(max_iter=100)
            elif model=="LR":
                clf.set_params(max_iter=100)
            if model in ["lasso","RFR","lgbm_reg"]:
                scoring="RAE_comp"
            else:
                scoring='f1_micro'

            #####
            ##### The AutoFE process is performed below with the iife() function
            #####	

            X_train, y_train, X_test, vartype_list, cvs, operation_list = iife(data_input = data_input, model = model, clf=clf, scoring=scoring, K=3, task = task, patience=20 if not task in ["cal_housing","fri"] else 40, int_inf_subset = 3000, eps=0, simul=False, seed=seed, seed2=seed2)

            #####
            ##### Write intermediate outputs to file
            #####

            f = open(output_dir + f"iife_validation_per_iter_{task}_{model}_{seed}_{seed2}.txt", "w")
            f.write(str(cvs))
            f.close()

            f = open(output_dir + f"iife_operation_list_{task}_{model}_{seed}_{seed2}.txt", "w")
            f.write(str(operation_list))
            f.close()

            #####
            ##### Hyperparameter tune AGAIN after AutoFE
            #####
            data=(X_train,y_train,X_test, y_test,feat_list,X_orig,vartype_list)
            hyperparams, hyperscores = hyperparam_tune(task,model,seed, data=data)
            hyperparams_params_after.append(hyperparams)

            if model == "LR":
                clf = LogisticRegression(C=hyperparams["C"],max_iter=50000)
            elif model == "RFR":
                clf = RandomForestRegressor(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
            elif model == "RF":
                clf = RandomForestClassifier(max_depth=hyperparams["max_depth"], max_features= hyperparams["max_features"], max_samples= hyperparams["max_samples"], n_estimators=hyperparams["n_estimators"], random_state=0)
            elif model == "lasso":
                clf = Lasso(alpha=hyperparams["alpha"],max_iter=500000)
            elif model == "lgbm_reg":
                clf = lgb.LGBMRegressor(
                    n_estimators = hyperparams["n_estimators"],
                    learning_rate = hyperparams["learning_rate"],
                    subsample = hyperparams["subsample"],
                    colsample_bytree = hyperparams["colsample_bytree"],
                    reg_lambda = hyperparams["reg_lambda"], 
                    random_state=0, num_threads=1
                )
            elif model == "lgbm_class":
                clf = lgb.LGBMClassifier(
                    n_estimators = hyperparams["n_estimators"],
                    learning_rate = hyperparams["learning_rate"],
                    subsample = hyperparams["subsample"],
                    colsample_bytree = hyperparams["colsample_bytree"],
                    reg_lambda = hyperparams["reg_lambda"], 
                    random_state=0, num_threads=1
                )


            if model == "LR" or model == "lasso":
                if "cat" in vartype_list:
                    # one hot encode
                    enc = OneHotEncoder(handle_unknown='ignore')
                    X_train_num = X_train[:,np.where(np.array(vartype_list)=="num")[0]]
                    X_train_cat = X_train[:,np.where(np.array(vartype_list)=="cat")[0]]
                    X_test_num = X_test[:,np.where(np.array(vartype_list)=="num")[0]]
                    X_test_cat = X_test[:,np.where(np.array(vartype_list)=="cat")[0]]
                    enc.fit(X_train_cat)
                    X_train_ohe = enc.transform(X_train_cat).toarray()
                    X_test_ohe = enc.transform(X_test_cat).toarray()

                    X_train_temp = np.concatenate((X_train_num,X_train_ohe),axis=1)
                    X_test_temp = np.concatenate((X_test_num,X_test_ohe),axis=1)
                else:
                    X_train_temp=X_train
                    X_test_temp=X_test

                scaler = MinMaxScaler()
                scaler.fit(X_train_temp)
                X_train_temp=scaler.transform(X_train_temp)
                X_test_temp=scaler.transform(X_test_temp)
            else:
                X_train_temp=X_train
                X_test_temp=X_test

            #####
            ##### Find test scores after AutoFE
            #####

            clf.fit(X_train_temp, y_train)
            pred = clf.predict(X_test_temp)

            if model=="LR":
                loss_after.append(log_loss(y_train,clf.predict_proba(X_train_temp)) ) #+ np.linalg.norm(clf.coef_,2)

            if model in ["lasso","RFR","lgbm_reg"]:
                score = RAE_comp(y_test,pred)
            else:
                score = f1_score(y_test,pred,average="micro")
            scores_after.append(score)
            end1=time.time()
            times.append(end1-start1)


            logging.info(f"Score before AutoFE: {scores_before[-1]}")
            logging.info(f"Score before AutoFE: {scores_after[-1]}")
            logging.info(f"AutoFE Time spent: {times[-1]}")

            #####
            ##### Store results of this runs
            #####
            f = open(output_dir + f"iife_{task}_{model}_{seed}_{seed2}.txt", "w")
            f.write("Baseline test score: " + str(scores_before[-1]))
            f.write("\n Transformed test score: " + str(scores_after[-1]))
            f.write("\n Time: " + str(times[-1]))
            f.close()


            f = open(output_dir + f"iife_vartype_list_{task}_{model}_{seed}_{seed2}.txt", "w")
            f.write(str(vartype_list))
            f.close()

            with open(output_dir + f'iife_X_train_{task}_{model}_{seed}_{seed2}.npy', 'wb') as f:
                np.save(f, X_train)


            with open(output_dir + f'iife_X_test_{task}_{model}_{seed}_{seed2}.npy', 'wb') as f:
                np.save(f, X_test)

    logging.info(f"Scores before AutoFE: {scores_before}")
    logging.info(f"Scores before AutoFE: {scores_after}")
    logging.info(f"AutoFE Time spent: {times}")

    #####
    ##### Store the results of the entire run
    #####
    f = open(output_dir + f"iife_{task}_{model}_FULL_run.txt", "w")
    f.write("Baseline test scores: " + str(scores_before))
    f.write("\n Transformed test scores: " + str(scores_after))
    f.write("\n Times: " + str(times))
    f.close()
    scores_before=[]
    scores_after=[]
    times=[]




    ### NEW: 把全局最优序列保存下来（你已有）
    f = open(output_dir + f"iife_operation_list_{task}_{model}_{seed}_{seed2}.txt", "w")
    f.write(str(operation_list))
    f.close()

    ### NEW: 对 4 个子集做同样的评估，统计均值/方差
    client_ids = [1, 2, 3, 4]
    client_scores_before, client_scores_after = [], []

    for cid in client_ids:
        # 1) 取子集
        Xc_tr, yc_tr, Xc_te, yc_te, feat_list_c, Xorig_c, vtypes_c = load_openml586_client_subset(seed, cid)
        yc_tr = np.array(yc_tr); yc_te = np.array(yc_te)

        # 2) （可选）重新做一次超参调优（子集数据上）
        data_c = (Xc_tr, yc_tr, Xc_te, yc_te, feat_list_c, Xorig_c, vtypes_c)
        hyper_c, _ = hyperparam_tune("openml586", model, seed, data=data_c)
        clf_c = _make_clf(model, hyper_c)

        # 3) Baseline（AutoFE 前）
        Xc_tr_tmp, Xc_te_tmp = _prep_inputs_for_model(model, Xc_tr, Xc_te, vtypes_c)
        clf_c.fit(Xc_tr_tmp, yc_tr)
        pred_c_base = clf_c.predict(Xc_te_tmp)
        if model in ["lasso", "RFR", "lgbm_reg"]:
            base_score = RAE_comp(yc_te, pred_c_base)
        else:
            base_score = f1_score(yc_te, pred_c_base, average="micro")
        client_scores_before.append(base_score)

        # 4) AutoFE（在该子集上单独运行一遍 iife）
        if model in ["lasso", "LR"]:
            # 降低迭代以加速（跟你全局逻辑一致）
            if hasattr(clf_c, "set_params"):
                clf_c.set_params(max_iter=100)
        scoring = "RAE_comp" if model in ["lasso", "RFR", "lgbm_reg"] else "f1_micro"
        Xc_tr_fe, yc_tr_fe, Xc_te_fe, vtypes_c_fe, cvs_c, ops_c = iife(
            data_input=data_c, model=model, clf=clf_c, scoring=scoring,
            K=3, task="openml586", patience=20, int_inf_subset=3000, eps=0,
            simul=False, seed=seed, seed2=seed2
        )

        # 5) AutoFE 后再调参+测试
        hyper_c2, _ = hyperparam_tune("openml586", model, seed, data=(Xc_tr_fe, yc_tr_fe, Xc_te_fe, yc_te, feat_list_c, Xorig_c, vtypes_c_fe))
        clf_c2 = _make_clf(model, hyper_c2)
        Xc_tr_tmp2, Xc_te_tmp2 = _prep_inputs_for_model(model, Xc_tr_fe, Xc_te_fe, vtypes_c_fe)
        clf_c2.fit(Xc_tr_tmp2, yc_tr_fe)
        pred_c_after = clf_c2.predict(Xc_te_tmp2)
        if model in ["lasso", "RFR", "lgbm_reg"]:
            aft_score = RAE_comp(yc_te, pred_c_after)
        else:
            aft_score = f1_score(yc_te, pred_c_after, average="micro")
        client_scores_after.append(aft_score)

    # 6) 汇总：均值/方差
    import numpy as np
    client_mean_before = float(np.mean(client_scores_before))
    client_var_before  = float(np.var (client_scores_before, ddof=1))  # 样本方差
    client_mean_after  = float(np.mean(client_scores_after))
    client_var_after   = float(np.var (client_scores_after,  ddof=1))

    # 7) 写到文件里
    with open(output_dir + f"iife_openml586_{model}_{seed}_{seed2}_CLIENT_STATS.txt", "w") as f:
        f.write("Global best operation_list:\n")
        f.write(str(operation_list) + "\n\n")
        f.write(f"Client scores BEFORE AutoFE: {client_scores_before}\n")
        f.write(f"Client scores AFTER  AutoFE: {client_scores_after}\n")
        f.write(f"Mean/Var BEFORE: {client_mean_before:.6f} / {client_var_before:.6f}\n")
        f.write(f"Mean/Var AFTER : {client_mean_after:.6f} / {client_var_after:.6f}\n")