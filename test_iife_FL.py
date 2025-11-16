#test_iife_FL.py

import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

# ----------- 基本类型 -----------

Pair = Tuple[int, int]



import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.special import expit


###################################################
#环境构建
###################################################
#定义算子表
def cube(x):
    return x ** 3
def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
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
        o = np.log
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
###################################################


# ----------- 交互信息（本地） + Fed-II 聚合 -----------

Pair = Tuple[int, int]
def interaction_mi(fi, fj, y, n_neighbors=3, random_state=0):
    """
    计算交互信息: I(fi; fj; y) = I(fi,fj; y) - I(fi; y) - I(fj; y)
    """
    # I(fi,fj; y)
    X_pair = np.column_stack([fi, fj])
    I_pair_y = mutual_info_classif(
        X_pair, y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    # I(fi; y)
    I_fi_y = mutual_info_classif(
        fi.reshape(-1, 1), y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    # I(fj; y)
    I_fj_y = mutual_info_classif(
        fj.reshape(-1, 1), y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    # 交互信息
    return float(I_pair_y - I_fi_y - I_fj_y)


def compute_local_tau(
    X_c: np.ndarray,
    y_c: np.ndarray,
    pairs: List[Pair]
) -> Dict[Pair, float]:
    """
    在单个 client 上，对给定的特征对列表 pairs，计算 τ_ij^c。
    """
    stats = {}
    for (i, j) in pairs:
        fi = X_c[:, i]
        fj = X_c[:, j]
        tau = interaction_mi(fi, fj, y_c)
        stats[(i, j)] = tau
    return stats

def fed_ii(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    pairs: List[Pair],
    weights: np.ndarray = None
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
        compute_local_tau(X_c, y_c, pairs)
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
    ):
        self.clients_data = [(X.copy(), y.copy()) for (X, y) in clients_data]
        self.task_type = task_type
        self.max_rounds = max_rounds
        self.top_k_pairs = top_k_pairs
        self.patience = patience
        self.weights = weights

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
        best_score = fed_eval_task(self.clients_data, self.task_type, self.weights)
        self.score_history.append(best_score)
        no_improve = 0

        for rnd in range(self.max_rounds):
            print(f"[Round {rnd}] current best federated score = {best_score:.4f}")

            # 1) Fed-II：当前特征对的全局交互信息
            pairs = self._all_pairs()
            I_global = fed_ii(self.clients_data, pairs, self.weights)
            sorted_pairs = sorted(I_global.items(), key=lambda kv: kv[1], reverse=True)
            top_pairs = [p for (p, _) in sorted_pairs[: self.top_k_pairs]]

            # 2) 在 Top-K 对上搜索 (binary_op, unary_op)
            cand_best_score = -np.inf
            cand_best_op = None

            for (i, j) in top_pairs:
                for b in self.binary_ops:
                    for u in self.unary_ops:
                        tmp_clients = []
                        for (X_c, y_c) in self.clients_data:
                            fi = X_c[:, i]
                            fj = X_c[:, j]
                            g = apply_binary_op(b, fi, fj)
                            new_feat = apply_unary_op(u, g).reshape(-1, 1)
                            X_tmp = np.concatenate([X_c, new_feat], axis=1)
                            tmp_clients.append((X_tmp, y_c))

                        s = fed_eval_task(tmp_clients, self.task_type, self.weights)
                        if s > cand_best_score:
                            cand_best_score = s
                            cand_best_op = Operation(b, u, i, j)

            print(f"  best candidate score this round = {cand_best_score:.4f}")

            # 早停逻辑
            if cand_best_score <= best_score + 1e-8:
                no_improve += 1
                print(f"  no improvement, patience {no_improve}/{self.patience}")
                if no_improve >= self.patience:
                    print("Early stopping.")
                    break
                else:
                    continue
            else:
                no_improve = 0

            # 3) commit 最佳操作：真正加到所有 client 的特征池
            op = cand_best_op
            self.operation_list.append(op)

            new_clients = []
            for (X_c, y_c) in self.clients_data:
                fi = X_c[:, op.i]
                fj = X_c[:, op.j]
                g = apply_binary_op(op.binary_op, fi, fj)
                new_feat = apply_unary_op(op.unary_op, g).reshape(-1, 1)
                X_new = np.concatenate([X_c, new_feat], axis=1)
                new_clients.append((X_new, y_c))
            self.clients_data = new_clients

            best_score = cand_best_score
            self.score_history.append(best_score)

        self.best_score_ = best_score
        return self


# ----------- 一个 demo：造数据 + 跑一遍 -----------

# ---------- 一个小 demo：分类任务用 F1 ----------
def demo_cls():
    # 造一个二分类数据集
    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )

    splits = np.array_split(np.arange(X.shape[0]), 3)
    clients = [(X[idx], y[idx]) for idx in splits]
    weights = np.array([len(idx) for idx in splits], dtype=float)

    fed_iife = FedIIFE(
        clients_data=clients,
        task_type='cls',
        max_rounds=3,
        top_k_pairs=5,
        patience=2,
        weights=weights,
    )
    fed_iife.fit()

    print("\n[CLS] Final best federated score (F1):", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")


# ---------- 一个小 demo：回归任务用 1-RAE ----------

def demo_reg():
    X, y = make_regression(
        n_samples=600,
        n_features=8,
        n_informative=5,
        noise=0.5,
        random_state=42,
    )

    splits = np.array_split(np.arange(X.shape[0]), 3)
    clients = [(X[idx], y[idx]) for idx in splits]
    weights = np.array([len(idx) for idx in splits], dtype=float)

    fed_iife = FedIIFE(
        clients_data=clients,
        task_type='reg',
        max_rounds=3,
        top_k_pairs=5,
        patience=2,
        weights=weights,
    )
    fed_iife.fit()

    print("\n[REG] Final best federated score (1-RAE):", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.unary_op}({op.binary_op}(F{op.i}, F{op.j}))")


if __name__ == "__main__":
    demo_reg()
