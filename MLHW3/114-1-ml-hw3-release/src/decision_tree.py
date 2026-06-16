import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):

        self.max_depth = max_depth
        self.feature_importance = None

    def fit(self, X, y):

        n_features = X.shape[1]
        self.feature_importance = np.zeros(n_features)
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):

        label_class = np.unique(y)


        if len(label_class) == 1 or depth >= self.max_depth:

            leaf_label = self.max_counts_label(y)
            return {"if_leaf": True, "label": leaf_label}

        
        best_feature_idx, best_threshold, best_gain = find_best_split(X, y)


        if best_feature_idx is None:
            leaf_label = self.max_counts_label(y)
            return {"if_leaf": True, "label": leaf_label}

        
        self.feature_importance[best_feature_idx] += best_gain

        R_idxs = X[:, best_feature_idx] > best_threshold
        L_idxs = X[:, best_feature_idx] <= best_threshold
        

        R_subtree = self._grow_tree(X[R_idxs], y[R_idxs], depth + 1)
        L_subtree = self._grow_tree(X[L_idxs], y[L_idxs], depth + 1)
        

        return {
            "if_leaf": False,
            "feature_index": best_feature_idx,
            "threshold": best_threshold,
            "R_subtree": R_subtree,
            "L_subtree": L_subtree,
        }

        raise NotImplementedError

    def predict(self, X):

        return np.array([self._predict_tree(x, self.tree) for x in X])
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        
        if tree_node["if_leaf"]:
            return tree_node["label"]

        feature_index = tree_node["feature_index"]
        threshold = tree_node["threshold"]

        if x[feature_index] > threshold:
            return self._predict_tree(x, tree_node["R_subtree"])
        else:
            return self._predict_tree(x, tree_node["L_subtree"])

        raise NotImplementedError
    

    def max_counts_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
     
    R_idxs = X[:, feature_index] > threshold
    L_idxs = X[:, feature_index] <= threshold
    
    return y[R_idxs], y[L_idxs]
    raise NotImplementedError


# Find the best split for the dataset
def find_best_split(X, y):

    n_samples, n_features = X.shape

    if n_samples <= 1:
        return None, None

    root_entropy = entropy(y)
    
    best_feature_idx = None
    best_threshold = None
    best_gain = 0.0

    for feature_index in range(n_features):

        thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            y_L, y_R = split_dataset(X, y, feature_index, threshold)

            if len(y_L) == 0 or len(y_R) == 0:
                continue

            
            L_entropy = entropy(y_L)
            R_entropy = entropy(y_R)

            n_all = len(y)
            n_L = len(y_L)
            n_R = len(y_R)

            split_entropy = float(n_L / n_all) * L_entropy + float(n_R / n_all) * R_entropy

            gain = root_entropy - split_entropy

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_index
                best_threshold = threshold

    return best_feature_idx, best_threshold, best_gain
    raise NotImplementedError


def entropy(y):

    n_all = len(y)

    if n_all == 0:
        return 0
    
    _, counts = np.unique(y, return_counts=True)

    prob = counts / n_all

    return -np.sum(prob * np.log2(prob+ 1e-10))
    raise NotImplementedError


def gini_index(y):

    n_all = len(y)

    if n_all == 0:
        return 0
    

    _, counts = np.unique(y, return_counts=True)

    p = (counts / n_all)**2

    
    return 1 - np.sum(p)
    raise NotImplementedError