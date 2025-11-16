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





# ----------- 交互信息（本地） + Fed-II 聚合 -----------

def interaction_mi(fi, fj, y, n_neighbors=3, random_state=0):
    """
    非常粗糙的交互信息近似：
    I([fi,fj]; y)  （这里只算这一项，忽略 I(fi;fj)，目的是简化示例）
    """
    X_pair = np.column_stack([fi, fj])
    # mutual info between [fi,fj] and y
    vals = mutual_info_classif(
        X_pair, y,
        discrete_features="auto",
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    # 这里简单取第一维；更严谨可以求和或做别的处理
    return float(vals[0])


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


# ----------- 联邦下游模型评估 -----------

def make_fixed_model(random_state=0):
    """
    固定的下游模型，不调参。
    """
    return LogisticRegression(max_iter=500, solver="lbfgs", random_state=random_state)


def local_eval_score(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
    cv_splits: int = 3
) -> float:
    """
    单个 client 上，用 K-fold CV 算 F1_micro。
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf = model_factory()
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], y_pred, average="micro"))
    return float(np.mean(scores))


def fed_eval(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], object],
    weights: np.ndarray = None,
    cv_splits: int = 3
) -> float:
    """
    联邦评估：各 client 本地做 CV，score 做加权平均。
    """
    Q = len(clients_data)
    if weights is None:
        weights = np.ones(Q) / Q
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    local_scores = []
    for (X_c, y_c) in clients_data:
        s = local_eval_score(X_c, y_c, model_factory, cv_splits=cv_splits)
        local_scores.append(s)

    local_scores = np.asarray(local_scores, dtype=float)
    return float(np.sum(weights * local_scores))


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



# ----------- Fed-IIFE 主循环 -----------

class FedIIFE:
    def __init__(self,
                 clients_data,
                 model_factory,
                 max_rounds=5,
                 top_k_pairs=10,
                 patience=3,
                 weights=None,
                 binary_ops=None,
                 unary_ops=None):
        self.clients_data = [(X.copy(), y.copy()) for (X, y) in clients_data]
        self.model_factory = model_factory
        self.max_rounds = max_rounds
        self.top_k_pairs = top_k_pairs
        self.patience = patience
        self.weights = weights

        # 用你的算子名
        self.binary_ops = binary_ops or ['+', '-', '*', '/']
        self.unary_ops  = unary_ops  or [
            'sqrt', 'square', 'sin', 'cos', 'tanh',
            'reciprocal', 'exp', 'cube', 'sigmoid', 'log',
            'stand_scaler', 'minmax_scaler', 'quan_trans'
        ]

        self.operation_list: List[Operation] = []
        self.score_history: List[float] = []


    # 当前特征索引 / 全部 pair
    def _current_feature_indices(self):
        d = self.clients_data[0][0].shape[1]
        return list(range(d))

    def _all_pairs(self):
        idx = self._current_feature_indices()
        return list(combinations(idx, 2))

    def fit(self):
        # 初始联邦 score（只用原始特征）
        best_score = fed_eval(self.clients_data, self.model_factory, self.weights)
        self.score_history.append(best_score)
        no_improve_rounds = 0

        for round_id in range(self.max_rounds):
            print(f"Round {round_id}, current best federated score = {best_score:.4f}")

            # 1) Fed-II：对当前所有特征对算全局交互信息
            pairs = self._all_pairs()
            I_global = fed_ii(self.clients_data, pairs, self.weights)

            # 选出交互信息最大的 Top-K 对
            sorted_pairs = sorted(I_global.items(), key=lambda kv: kv[1], reverse=True)
            top_pairs = [p for (p, _) in sorted_pairs[: self.top_k_pairs]]

            # 2) 对每个 (pair, op) 作为候选新特征做联邦评估
            cand_best_score = -np.inf
            cand_best_op = None

            for (i, j) in top_pairs:
                for b in self.binary_ops:
                    for u in self.unary_ops:
                        tmp_clients = []
                        for (X_c, y_c) in self.clients_data:
                            fi = X_c[:, i]
                            fj = X_c[:, j]
                            g = apply_binary_op(b, fi, fj)       # 先二元算子
                            new_feat = apply_unary_op(u, g).reshape(-1, 1)  # 再一元算子
                            X_tmp = np.concatenate([X_c, new_feat], axis=1)
                            tmp_clients.append((X_tmp, y_c))

                        s = fed_eval(tmp_clients, self.model_factory, self.weights)
                        if s > cand_best_score:
                            cand_best_score = s
                            cand_best_op = Operation(b, u, i, j)


            print(f"  best candidate score this round = {candidate_best_score:.4f}")

            # 早停判断
            if candidate_best_score <= best_score:
                no_improve_rounds += 1
                print(f"  no improvement, patience {no_improve_rounds}/{self.patience}")
                if no_improve_rounds >= self.patience:
                    print("Early stopping.")
                    break
                else:
                    continue
            else:
                no_improve_rounds = 0

            # 3) commit 最佳候选：把新特征真正加进所有 client 的特征池
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


        self.best_score_ = best_score
        return self


# ----------- 一个 demo：造数据 + 跑一遍 -----------

def demo():
    # 造一个二分类数据集
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
    )

    # 按行切成 3 个 client
    splits = np.array_split(np.arange(X.shape[0]), 3)
    clients = [(X[idx], y[idx]) for idx in splits]
    weights = np.array([len(idx) for idx in splits], dtype=float)

    fed_iife = FedIIFE(
        clients_data=clients,
        model_factory=lambda: make_fixed_model(random_state=0),
        max_rounds=3,     # 为了 demo，轮数开小一点
        top_k_pairs=5,    # 每轮只在 5 个最强交互对上搜索
        patience=2,       # 2 轮不提升就停
        weights=weights,
    )
    fed_iife.fit()

    print("\nFinal best federated score:", fed_iife.best_score_)
    print("Operation sequence:")
    for k, op in enumerate(fed_iife.operation_list, 1):
        print(f"  step {k}: new_feat = {op.op_name}(F{op.i}, F{op.j})")


if __name__ == "__main__":
    demo()
