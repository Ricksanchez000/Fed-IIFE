#test_iife_FL.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error,
    average_precision_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_regression

# ----------- 基本类型 -----------

Pair = Tuple[int, int]

from dataclasses import dataclass

@dataclass
class Operation:
    binary_op: str   # '+', '-', '*', '/'
    unary_op: str    # 'sqrt', 'stand_scaler', 'log', ...
    i: int           # feature index i
    j: int           # feature index j


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.special import expit


###################################################
#环境构建
###################################################
#定义算子表
def cube(x):
    return x ** 3

EPS = 1e-6

def safe_sqrt(x):
    # 把负数裁成 0，再开方
    return np.sqrt(np.clip(x, 0.0, None))

def safe_log(x):
    # 对 |x| 加一个很小的偏移，再 log；避免 log(0) 和 log(负数)
    return np.log(np.abs(x) + EPS)

def safe_reciprocal(x):
    # 避免除 0：把特别小的数抬到 ±EPS
    x = x.copy()
    mask = np.abs(x) < EPS
    x[mask] = EPS * np.sign(x[mask] + 1e-12)
    return 1.0 / x

def justify_operation_type(o):
    if o == 'sqrt':
        #o = np.sqrt
        o = safe_sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        #o = np.reciprocal
        o = safe_reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        #o = np.log
        o = safe_log
    else:
        raise ValueError(f'Unknown op {o}')
    return o

# ---------- 你的误差 + 下游测试函数 ----------
def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(
        np.abs(np.mean(y_test) - y_test)
    )
    return error

def test_task_new(Dg, task='cls', state_num=10):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]

    if task == 'cls':
        y = y.astype(int)
        clf = RandomForestClassifier(random_state=0)
        acc_list, pre_list, rec_list, f1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train = X.iloc[train, :], y.iloc[train]
            X_test, y_test = X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_predict))
            pre_list.append(
                precision_score(
                    y_test, y_predict,
                    average='weighted', zero_division=0
                )
            )
            rec_list.append(
                recall_score(
                    y_test, y_predict,
                    average='weighted', zero_division=0
                )
            )
            f1_list.append(
                f1_score(
                    y_test, y_predict,
                    average='weighted', zero_division=0
                )
            )
        return np.mean(acc_list), np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)

    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train = X.iloc[train, :], y.iloc[train]
            X_test, y_test = X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(mean_absolute_error(y_test, y_predict))
            mse_list.append(mean_squared_error(y_test, y_predict))
            rae_list.append(relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)

    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        for train, test in kf.split(X):
            X_train, y_train = X.iloc[train, :], y.iloc[train]
            X_test, y_test = X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='macro'))
            ras.append(roc_auc_score(y_test, y_predict))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras)

    elif task == 'rank':
        # 先空着
        return -1
    else:
        return -1

# 给 Fed-IIFE 用的“标量打分”：cls→F1，reg→1-RAE
def local_eval_score_task(X: np.ndarray, y: np.ndarray, task: str) -> float:
    Dg = pd.DataFrame(X.copy())
    Dg['label'] = y
    if task == 'cls':
        _, _, _, f1 = test_task_new(Dg, 'cls')
        return float(f1)
    elif task == 'reg':
        _, _, rae = test_task_new(Dg, 'reg')
        return float(1.0 - rae)   # 越大越好
    else:
        raise ValueError(f"Unsupported task type: {task}")


def fed_eval_task(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    task: str,
    weights: np.ndarray = None,
) -> float:
    """
    联邦评估：每个 client 用 test_task_new 算本地 F1 或 1-RAE，再按样本数加权平均。
    """
    Q = len(clients_data)
    if weights is None:
        weights = np.ones(Q) / Q
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    local_scores = []
    for (X_c, y_c) in clients_data:
        s = local_eval_score_task(X_c, y_c, task)
        local_scores.append(s)

    local_scores = np.asarray(local_scores, dtype=float)
    return float(np.sum(weights * local_scores))

def sanitize_array(X, max_abs=1e6):
    """
    把 NaN / Inf / 极端大值裁剪到 [-max_abs, max_abs] 区间内，
    避免 RF / MI 计算时炸掉。
    """
    X = np.nan_to_num(X, nan=0.0, posinf=max_abs, neginf=-max_abs)
    X = np.clip(X, -max_abs, max_abs)
    return X


def apply_operation_sequence(X: np.ndarray, operation_list: List[Operation]) -> np.ndarray:
    """
    给定原始特征 X 和一串 FedIIFE 的 Operation，
    在同一个特征空间上按顺序重放变换，返回扩展后的特征矩阵。
    """
    X_new = X.copy()
    for op in operation_list:
        fi = X_new[:, op.i]
        fj = X_new[:, op.j]
        g = apply_binary_op(op.binary_op, fi, fj)
        new_feat = apply_unary_op(op.unary_op, g)
        new_feat = sanitize_array(new_feat).reshape(-1, 1)
        X_new = np.concatenate([X_new, new_feat], axis=1)
    return X_new


###################################################


# ----------- 交互信息（本地） + Fed-II 聚合 -----------

Pair = Tuple[int, int]
def interaction_mi(fi, fj, y, task_type='cls', n_neighbors=3, random_state=0):
    """
    计算交互信息: I(fi; fj; y) = I(fi,fj; y) - I(fi; y) - I(fj; y)
    
    Args:
        task_type: 'cls' 用 mutual_info_classif, 'reg' 用 mutual_info_regression
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # 根据任务选择互信息函数
    if task_type == 'cls':
        mi_func = mutual_info_classif
    elif task_type == 'reg':
        mi_func = mutual_info_regression
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    X_pair = np.column_stack([fi, fj])
    
    # I(fi,fj; y)
    mi_pair = mi_func(
        X_pair, y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    I_pair_y = float(mi_pair.sum())
    
    # I(fi; y)
    I_fi_y = mi_func(
        fi.reshape(-1, 1), y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    # I(fj; y)
    I_fj_y = mi_func(
        fj.reshape(-1, 1), y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    return float(I_pair_y - I_fi_y - I_fj_y)


from knncmi import cmi as knn_cmi
import pandas as pd

def interaction_ii_knn(fi, fj, y, k=4, discrete_dist=1, minzero=1):
    """
    用 knncmi 的 kNN 估计器计算论文里的交互信息：
        τ_ij = I(Fi; Fj | Y) - I(Fi; Fj)

    参数：
        fi, fj: 一维 np.ndarray，形状 (n,)
        y     : 一维 np.ndarray，形状 (n,)
        k     : k-NN 的 k
        discrete_dist, minzero: 直接透传给 knncmi.cmi

    返回：
        float 标量 τ_ij
    """
    # 组一个 DataFrame，方便按列名调用 knn_cmi
    df = pd.DataFrame({
        "fi": fi,
        "fj": fj,
        "y":  y,
    })

    # I(Fi; Fj | Y)
    I_cond = knn_cmi(
        x=["fi"],
        y=["fj"],
        z=["y"],
        k=k,
        data=df,
        discrete_dist=discrete_dist,
        minzero=minzero,
    )

    # I(Fi; Fj)
    I_marg = knn_cmi(
        x=["fi"],
        y=["fj"],
        z=[],          # 没有条件变量 → 退化成普通 MI
        k=k,
        data=df,
        discrete_dist=discrete_dist,
        minzero=minzero,
    )

    return float(I_cond - I_marg)



def compute_local_tau(
    X_c: np.ndarray,
    y_c: np.ndarray,
    pairs: List[Pair],
    task_type='cls'
) -> Dict[Pair, float]:
    """
    在单个 client 上，对给定的特征对列表 pairs，计算 τ_ij^c。
    """
    stats = {}
    for (i, j) in pairs:
        fi = X_c[:, i]
        fj = X_c[:, j]
        #tau = interaction_mi(fi, fj, y_c, task_type=task_type)
        tau = interaction_ii_knn(fi, fj, y_c, k=4)
        stats[(i, j)] = tau
    return stats

def fed_ii(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    pairs: List[Pair],
    weights: np.ndarray = None,
    task_type='cls'
) -> Dict[Pair, float]:
    """
    Fed-II：对所有 client 的本地 τ_ij^c 做（加权）平均，得到全局 τ_ij。
    返回 dict: {(i,j): τ_ij}
    """
    Q = len(clients_data)
    if weights is None:
        weights = np.ones(Q) / Q
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # 各 client 本地计算 τ_ij^c
    local_stats = [
        compute_local_tau(X_c, y_c, pairs, task_type=task_type)
        for (X_c, y_c) in clients_data
    ]

    # 服务器聚合
    I_global = {}
    for p in pairs:
        vals = np.array([stats[p] for stats in local_stats], dtype=float)
        I_global[p] = float(np.sum(weights * vals))
    return I_global




# ----------- 特征构造算子 -----------
def apply_binary_op(op_name: str, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    """
    二元算子 b(F_i, F_j)，只使用 '+', '-', '*', '/' 这类操作。
    """
    op = justify_operation_type(op_name)
    # 这里假设二元算子都是 numpy 的 ufunc（不带 fit_transform）
    if callable(op) and not hasattr(op, "fit_transform"):
        return op(fi, fj)
    raise ValueError(f"Operator {op_name} not suitable as binary op")

def apply_unary_op(op_name: str, f: np.ndarray) -> np.ndarray:
    """
    一元算子 u(g)，可以是函数，也可以是 Scaler/QuantileTransformer。
    """
    op = justify_operation_type(op_name)
    if hasattr(op, "fit_transform"):        # StandardScaler / MinMaxScaler / Quantile
        return op.fit_transform(f.reshape(-1, 1)).ravel()
    else:                                   # np.sqrt / np.log / sigmoid / cube ...
        return op(f)

from dataclasses import dataclass

@dataclass
class Operation:
    binary_op: str   # '+', '-', '*', '/'
    unary_op: str    # 'sqrt', 'stand_scaler', 'log', ...
    i: int           # feature index i
    j: int           # feature index j



# ---------- Fed-IIFE 主循环：用 F1 / 1-RAE 做目标 ----------

# ---------- Fed-IIFE 主循环：用 F1 / 1-RAE 做目标 ----------

class FedIIFE:
    def __init__(
        self,
        clients_data: List[Tuple[np.ndarray, np.ndarray]],
        task_type: str,                      # 'cls' 或 'reg'
        max_rounds: int = 5,
        top_k_pairs: int = 10,
        patience: int = 3,
        weights: np.ndarray = None,
        binary_ops: List[str] = None,
        unary_ops: List[str] = None,
        verbose: int = 1,  # 新增：0=静默, 1=正常, 2=详细
    ):
        self.clients_data = [(sanitize_array(X.copy()), y.copy())
                            for (X, y) in clients_data]

        self.task_type = task_type
        self.max_rounds = max_rounds
        self.top_k_pairs = top_k_pairs
        self.patience = patience
        self.weights = weights
        self.verbose = verbose

        self.binary_ops = binary_ops or ['+', '-', '*', '/']
        self.unary_ops = unary_ops or [
            'sqrt', 'square', 'sin', 'cos', 'tanh',
            'reciprocal', 'exp', 'cube', 'sigmoid', 'log',
            'stand_scaler', 'minmax_scaler', 'quan_trans',
        ]

        self.operation_list: List[Operation] = []
        self.score_history: List[float] = []

    def _current_feature_indices(self):
        d = self.clients_data[0][0].shape[1]
        return list(range(d))

    def _all_pairs(self):
        idx = self._current_feature_indices()
        return list(combinations(idx, 2))

    def fit(self):
        import time
        
        # 初始信息
        n_clients = len(self.clients_data)
        n_features = self.clients_data[0][0].shape[1]
        total_samples = sum(len(y) for _, y in self.clients_data)
        
        if self.verbose >= 1:
            print("=" * 70)
            print(f"Fed-IIFE 初始化")
            print("=" * 70)
            print(f"客户端数量: {n_clients}")
            print(f"初始特征数: {n_features}")
            print(f"总样本数: {total_samples}")
            print(f"任务类型: {self.task_type}")
            print(f"最大轮数: {self.max_rounds}")
            print(f"Top-K pairs: {self.top_k_pairs}")
            print(f"Patience: {self.patience}")
            print(f"二元算子: {self.binary_ops}")
            print(f"一元算子数量: {len(self.unary_ops)}")
            print("=" * 70)
        
        # 初始评估
        if self.verbose >= 1:
            print("\n[初始评估] 计算基线性能...")
        
        start_time = time.time()
        best_score = fed_eval_task(self.clients_data, self.task_type, self.weights)
        eval_time = time.time() - start_time
        
        self.score_history.append(best_score)
        no_improve = 0

        if self.verbose >= 1:
            print(f"[初始评估] 基线得分: {best_score:.6f} (耗时: {eval_time:.2f}s)")
            print()

        for rnd in range(self.max_rounds):
            round_start = time.time()
            
            if self.verbose >= 1:
                print("=" * 70)
                print(f"Round {rnd + 1}/{self.max_rounds}")
                print("=" * 70)
                print(f"当前最佳得分: {best_score:.6f}")
                print(f"当前特征数: {self.clients_data[0][0].shape[1]}")

            # 1) Fed-II：计算交互信息
            if self.verbose >= 1:
                print(f"\n[步骤1] 计算特征对交互信息...")
            
            mi_start = time.time()
            pairs = self._all_pairs()
            
            if self.verbose >= 2:
                print(f"  - 总特征对数: {len(pairs)}")
            
            I_global = fed_ii(self.clients_data, pairs, self.weights, task_type=self.task_type)
            sorted_pairs = sorted(I_global.items(), key=lambda kv: kv[1], reverse=True)
            top_pairs = [p for (p, _) in sorted_pairs[: self.top_k_pairs]]
            mi_time = time.time() - mi_start
            
            if self.verbose >= 1:
                print(f"  - 耗时: {mi_time:.2f}s")
                print(f"  - Top-{self.top_k_pairs} 特征对交互信息:")
                for idx, (pair, score) in enumerate(sorted_pairs[:5], 1):
                    print(f"    {idx}. F{pair[0]} & F{pair[1]}: {score:.6f}")
                if len(sorted_pairs) > 5:
                    print(f"    ...")

            # 2) 搜索最佳操作组合
            if self.verbose >= 1:
                print(f"\n[步骤2] 搜索最佳特征变换...")
                total_combinations = len(top_pairs) * len(self.binary_ops) * len(self.unary_ops)
                print(f"  - 候选组合数: {total_combinations}")
            
            search_start = time.time()
            cand_best_score = -np.inf
            cand_best_op = None
            
            eval_count = 0
            for pair_idx, (i, j) in enumerate(top_pairs):
                if self.verbose >= 2:
                    print(f"\n  特征对 {pair_idx + 1}/{len(top_pairs)}: F{i} & F{j}")
                
                for b in self.binary_ops:
                    for u in self.unary_ops:
                        eval_count += 1
                        
                        try:
                            tmp_clients = []
                            for (X_c, y_c) in self.clients_data:
                                fi = X_c[:, i]
                                fj = X_c[:, j]
                                g = apply_binary_op(b, fi, fj)
                                new_feat = apply_unary_op(u, g).reshape(-1, 1)
                                new_feat = sanitize_array(new_feat).reshape(-1, 1)
                                X_tmp = np.concatenate([X_c, new_feat], axis=1)
                                tmp_clients.append((X_tmp, y_c))

                            s = fed_eval_task(tmp_clients, self.task_type, self.weights)
                            
                            if s > cand_best_score:
                                cand_best_score = s
                                cand_best_op = Operation(b, u, i, j)
                                
                                if self.verbose >= 2:
                                    print(f"    ✓ 新最佳: {u}({b}(F{i}, F{j})) = {s:.6f}")
                        
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"    ✗ 错误: {u}({b}(F{i}, F{j})) - {str(e)[:50]}")
                            continue
                        
                        # 进度显示
                        if self.verbose >= 1 and eval_count % 50 == 0:
                            print(f"    进度: {eval_count}/{total_combinations} ({100*eval_count/total_combinations:.1f}%)")
            
            search_time = time.time() - search_start
            
            if self.verbose >= 1:
                print(f"\n  - 评估完成: {eval_count}/{total_combinations} 个组合")
                print(f"  - 耗时: {search_time:.2f}s")
                print(f"  - 本轮最佳得分: {cand_best_score:.6f}")
                if cand_best_op:
                    print(f"  - 最佳操作: {cand_best_op.unary_op}({cand_best_op.binary_op}(F{cand_best_op.i}, F{cand_best_op.j}))")

            # 3) 早停判断
            improvement = cand_best_score - best_score
            
            if self.verbose >= 1:
                print(f"\n[步骤3] 早停检查")
                print(f"  - 性能提升: {improvement:.6f}")
            
            if cand_best_score <= best_score + 1e-8:
                no_improve += 1
                if self.verbose >= 1:
                    print(f"  - 无提升，patience计数: {no_improve}/{self.patience}")
                
                if no_improve >= self.patience:
                    if self.verbose >= 1:
                        print(f"  - 达到patience上限，早停！")
                    break
                else:
                    round_time = time.time() - round_start
                    if self.verbose >= 1:
                        print(f"\n本轮耗时: {round_time:.2f}s")
                    continue
            else:
                no_improve = 0
                if self.verbose >= 1:
                    print(f"  - 性能提升，继续训练")

            # 4) 提交最佳操作
            if self.verbose >= 1:
                print(f"\n[步骤4] 应用最佳特征变换")
            
            op = cand_best_op
            self.operation_list.append(op)

            commit_start = time.time()
            new_clients = []
            for (X_c, y_c) in self.clients_data:
                fi = X_c[:, op.i]
                fj = X_c[:, op.j]
                g = apply_binary_op(op.binary_op, fi, fj)
                new_feat = apply_unary_op(op.unary_op, g).reshape(-1, 1)
                new_feat = sanitize_array(new_feat).reshape(-1, 1)
                X_new = np.concatenate([X_c, new_feat], axis=1)
                new_clients.append((X_new, y_c))
            self.clients_data = new_clients
            commit_time = time.time() - commit_start

            best_score = cand_best_score
            self.score_history.append(best_score)
            
            round_time = time.time() - round_start
            
            if self.verbose >= 1:
                print(f"  - 特征添加完成，新特征数: {self.clients_data[0][0].shape[1]}")
                print(f"  - 应用耗时: {commit_time:.2f}s")
                print(f"\n本轮总耗时: {round_time:.2f}s")
                print(f"平均每个组合评估耗时: {search_time/eval_count:.4f}s")

        # 最终总结
        total_time = time.time() - start_time
        
        if self.verbose >= 1:
            print("\n" + "=" * 70)
            print("Fed-IIFE 训练完成")
            print("=" * 70)
            print(f"总轮数: {len(self.score_history) - 1}")
            print(f"最终特征数: {self.clients_data[0][0].shape[1]}")
            print(f"初始得分: {self.score_history[0]:.6f}")
            print(f"最终得分: {best_score:.6f}")
            print(f"性能提升: {best_score - self.score_history[0]:.6f}")
            print(f"总耗时: {total_time:.2f}s")
            print(f"\n特征变换序列:")
            for k, op in enumerate(self.operation_list, 1):
                print(f"  {k}. {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")
            print("=" * 70)

        self.best_score_ = best_score
        return self
    
    
###################################################
# 6. 读取 pima_indian 数据，并跑联邦 IIFE
###################################################
def load_pima_clients(data_dir: str, num_clients: int = 5):
    """
    从 data/ 目录读取 pima_indian_1.hdf ... pima_indian_5.hdf，
    每个文件一个 client，最后一列为标签。
    """
    clients = []
    for cid in range(1, num_clients + 1):
        path = os.path.join(data_dir, f"pima_indian_{cid}.hdf")
        df = pd.read_hdf(path).reset_index(drop=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        clients.append((X, y))
    weights = np.array([len(y) for (_, y) in clients], dtype=float)
    return clients, weights
def load_pima_global(data_dir: str):
    """
    读取 data/pima_indian.hdf 作为全局完整数据集。
    假设最后一列是标签。
    """
    path = os.path.join(data_dir, "pima_indian.hdf")
    df = pd.read_hdf(path).reset_index(drop=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y



def demo_pima_cls():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    # 1) 加载各个 client 子集
    clients, weights = load_pima_clients(data_dir, num_clients=5)

    # 2) 训练 Fed-IIFE（用联邦加权 F1 选特征池）
    fed_iife = FedIIFE(
        clients_data=clients,
        task_type='cls',
        max_rounds=10,
        top_k_pairs=5,
        patience=2,
        weights=weights,
        verbose=1,  # 需要详细过程再改成 2
    )
    fed_iife.fit()

    print("\n[PIMA CLS] Final best federated score (F1):", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")

    # =====================================================
    # A. 在全局完整数据上评估（pima_indian.hdf）
    # =====================================================
    Xg, yg = load_pima_global(data_dir)
    # 原始全局特征 + FedIIFE 的操作序列
    Xg_fe = apply_operation_sequence(Xg, fed_iife.operation_list)

    Dg_global = pd.DataFrame(Xg_fe)
    Dg_global['label'] = yg
    _, _, _, f1_global = test_task_new(Dg_global, 'cls')
    print("\n[Global full data] F1 after Fed-IIFE feature pool:", f1_global)

    # =====================================================
    # B. 在各个 client 上评估最终特征池，统计均值 / 标准差
    # =====================================================
    client_f1s = []
    for cid, (Xc, yc) in enumerate(fed_iife.clients_data, 1):
        Dc = pd.DataFrame(Xc)
        Dc['label'] = yc
        _, _, _, f1_c = test_task_new(Dc, 'cls')
        client_f1s.append(f1_c)
        print(f"  Client {cid} F1: {f1_c:.6f}")

    client_f1s = np.array(client_f1s)
    mean_f1 = client_f1s.mean()
    std_f1  = client_f1s.std(ddof=1)  # 样本标准差

    print("\n[Client stats]")
    print("  F1 per client:", client_f1s)
    print(f"  Mean F1      : {mean_f1:.6f}")
    print(f"  Std  F1      : {std_f1:.6f}")



num_clients = 4
def load_openml_586_clients(data_dir: str, num_clients: int = 4):
    """
    从 data/ 目录读取 openml_586_1.hdf ... openml_586_{num_clients}.hdf，
    每个文件一个 client，最后一列为标签。
    """
    clients = []
    for cid in range(1, num_clients + 1):
        path = os.path.join(data_dir, f"openml_586_{cid}.hdf")
        df = pd.read_hdf(path).reset_index(drop=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        clients.append((X, y))
    # 按样本数做加权
    weights = np.array([len(y) for (_, y) in clients], dtype=float)
    return clients, weights


def load_openml_586_global(data_dir: str):
    """
    读取 data/openml_586.hdf 作为全局完整数据集。
    假设最后一列是标签。
    """
    path = os.path.join(data_dir, "openml_586.hdf")
    df = pd.read_hdf(path).reset_index(drop=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def demo_586_reg():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    num_clients = 4

    # 1) 加载各个 client 子集
    clients, weights = load_openml_586_clients(data_dir, num_clients=num_clients)

    # 2) 训练 Fed-IIFE（用联邦加权 (1-RAE) 选特征池）
    fed_iife = FedIIFE(
        clients_data=clients,
        task_type='reg',         # 注意这里是回归
        max_rounds=3,
        top_k_pairs=5,
        patience=2,
        weights=weights,
        verbose=1,               # 需要更详细过程设成 2
    )
    fed_iife.fit()

    print("\n[OPENML-586 REG] Final best federated score (1-RAE):", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")

    # =====================================================
    # A. 在全局完整数据上评估（openml_586.hdf）
    # =====================================================
    Xg, yg = load_openml_586_global(data_dir)
    # 原始全局特征 + FedIIFE 的操作序列
    Xg_fe = apply_operation_sequence(Xg, fed_iife.operation_list)

    Dg_global = pd.DataFrame(Xg_fe)
    Dg_global['label'] = yg
    mae_g, mse_g, rae_g = test_task_new(Dg_global, 'reg')
    score_global = 1.0 - rae_g

    print("\n[Global full data] metrics after Fed-IIFE feature pool:")
    print(f"  MAE : {mae_g:.6f}")
    print(f"  MSE : {mse_g:.6f}")
    print(f"  RAE : {rae_g:.66f}")
    print(f"  1-RAE (score): {score_global:.6f}")

    # =====================================================
    # B. 在各个 client 上评估最终特征池，统计均值 / 标准差（按 1-RAE）
    # =====================================================
    client_scores = []
    for cid, (Xc, yc) in enumerate(fed_iife.clients_data, 1):
        Dc = pd.DataFrame(Xc)
        Dc['label'] = yc
        mae_c, mse_c, rae_c = test_task_new(Dc, 'reg')
        score_c = 1.0 - rae_c
        client_scores.append(score_c)
        print(f"  Client {cid} -> MAE={mae_c:.6f}, MSE={mse_c:.6f}, RAE={rae_c:.6f}, 1-RAE={score_c:.6f}")

    client_scores = np.array(client_scores)
    mean_score = client_scores.mean()
    std_score  = client_scores.std(ddof=1)  # 样本标准差

    print("\n[Client stats] (1-RAE as score)")
    print("  Scores per client:", client_scores)
    print(f"  Mean 1-RAE      : {mean_score:.6f}")
    print(f"  Std  1-RAE      : {std_score:.6f}")





###################################################
#  Wine Red a0p5: 联邦分类版本
###################################################

def load_wine_red_clients(data_dir: str, num_clients: int = 4):
    """
    从 data/ 目录读取 wine_red_a0p5_1.hdf ... wine_red_a0p5_{num_clients}.hdf，
    每个文件一个 client，最后一列为标签。
    """
    clients = []
    for cid in range(1, num_clients + 1):
        path = os.path.join(data_dir, f"wine_red_a0p5_{cid}.hdf")
        df = pd.read_hdf(path).reset_index(drop=True)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        clients.append((X, y))
    # 按样本数做加权
    weights = np.array([len(y) for (_, y) in clients], dtype=float)
    return clients, weights

def load_wine_red_global(data_dir: str):
    """
    读取 data/wine_red_a0p5.hdf 作为全局完整数据集。
    假设最后一列是标签。
    """
    path = os.path.join(data_dir, "wine_red_a0p5.hdf")
    df = pd.read_hdf(path).reset_index(drop=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def demo_wine_red_cls():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    num_clients = 4

    # 1) 加载各个 client 子集
    clients, weights = load_wine_red_clients(data_dir, num_clients=num_clients)

    # 2) 训练 Fed-IIFE（用联邦加权 F1 选特征池）
    fed_iife = FedIIFE(
        clients_data=clients,
        task_type='cls',        # wine_red 是分类
        max_rounds=5,
        top_k_pairs=5,
        patience=2,
        weights=weights,
        verbose=1,              # 想看更详细日志改成 2
    )
    fed_iife.fit()

    print("\n[WINE RED CLS] Final best federated score (F1):", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")

    # =====================================================
    # A. 在全局完整数据上评估（wine_red_a0p5.hdf）
    # =====================================================
    Xg, yg = load_wine_red_global(data_dir)
    Xg_fe = apply_operation_sequence(Xg, fed_iife.operation_list)

    Dg_global = pd.DataFrame(Xg_fe)
    Dg_global['label'] = yg
    _, _, _, f1_global = test_task_new(Dg_global, 'cls')
    print("\n[Global full data] F1 after Fed-IIFE feature pool:", f1_global)

    # =====================================================
    # B. 在各个 client 上评估最终特征池，统计均值 / 标准差（F1）
    # =====================================================
    client_f1s = []
    for cid, (Xc, yc) in enumerate(fed_iife.clients_data, 1):
        Dc = pd.DataFrame(Xc)
        Dc['label'] = yc
        _, _, _, f1_c = test_task_new(Dc, 'cls')
        client_f1s.append(f1_c)
        print(f"  Client {cid} F1: {f1_c:.6f}")

    client_f1s = np.array(client_f1s)
    mean_f1 = client_f1s.mean()
    std_f1  = client_f1s.std(ddof=1)  # 样本标准差

    print("\n[Client stats] (F1)")
    print("  F1 per client:", client_f1s)
    print(f"  Mean F1      : {mean_f1:.6f}")
    print(f"  Std  F1      : {std_f1:.6f}")

if __name__ == "__main__":
    demo_pima_cls()