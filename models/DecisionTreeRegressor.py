# Importing,
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class DecisionTreeRegressor():

    def __init__(self, max_depth):
        """Constructor method for the DecisionTreeClassifier class. We simply create the class variables."""

        # Class variables for the data and nodes,
        self.X, self.y, self.nodes, self.leaves = None, None, None, None

        # Stopping criteria,
        self.max_depth = max_depth

        return None
    
    def fit(self, X, y):

        # Assignment to class attributes,
        self.X, self.y = X, y

        # Creating the root node,
        root_node = Node(self.X, self.y, parent_node=None)
        self.nodes, self.leaves = [root_node], [root_node]

        # Growth algorithm,
        for i in range(self.max_depth):

            # Creating new layer,
            new_leaves = self._grow_tree()

            # Adding child nodes to list,
            self.nodes.extend(new_leaves)

        # Assigning predictions to leaves (terminate node),
        for leaf_node in self.leaves:
            leaf_node.prediction = np.mean(leaf_node.y)

    @staticmethod
    def _compute_SE(y_node):
        return np.sum((y_node - np.mean(y_node))**2)

    def _grow_tree(self):

        # Placeholder list,
        new_leaves = []

        # Looping through current leaves,
        for node in self.leaves:

            # Performing split,
            child_node_left, child_node_right, valid_split = self._split(node.X, node.y, parent_node=node)

            if valid_split:
                # Assigning child nodes,
                node.child_left, node.child_right, = child_node_left, child_node_right

                # Appending nodes to list,
                new_leaves.extend([child_node_left, child_node_right])
            else: 
                # If no split occured, the node remains a leaf,
                new_leaves.append(node)

        # Updating leaves,
        self.leaves = new_leaves

        return new_leaves

    def _split(self, X, y, parent_node):
        """Binary splits a parent node into two child nodes based on the decision that mimises the SSE (sum of squared errors) of the
        child nodes."""

        # Placeholder variables,
        min_SSE = np.inf
        split_threshold_value = None
        found_valid_split = False
        X_best_left_split, X_best_right_split, y_best_left_split, y_best_right_split = None, None, None, None

        # Double loop, first for each feature, second for each threshold value,
        for feature_idx in range(X.shape[1]):

            # Extracting feature values and thresholds,
            X_feature = X[:, feature_idx]
            thresholds = np.unique(X_feature)

            for threshold_value in thresholds:

                # Splitting data in parent node into child nodes,
                left_split_idxs, right_split_idxs = np.where(X_feature <= threshold_value)[0], np.where(X_feature > threshold_value)[0]
                X_left_split, X_right_split = X[left_split_idxs], X[right_split_idxs]
                y_left_split, y_right_split = y[left_split_idxs], y[right_split_idxs]

                # Reject splits which result in empty child nodes,
                if len(left_split_idxs) == 0 or len(right_split_idxs) == 0:
                    continue
                else:
                    found_valid_split = True

                # Compute SSE of child nodes,
                children_SSE = self._compute_SE(y_node=y_left_split) + self._compute_SE(y_node=y_right_split)
            
                # Tracking maximum information gain,
                if children_SSE < min_SSE:

                    # Updating nodes associated with the best split,
                    max_gain, split_threshold_value, split_feature = children_SSE, threshold_value, feature_idx
                    X_best_left_split, X_best_right_split, y_best_left_split, y_best_right_split = X_left_split, X_right_split, y_left_split, y_right_split

        # Creating node objects for the child nodes,
        if found_valid_split:
            child_node_left, child_node_right = Node(X_best_left_split, y_best_left_split, parent_node), Node(X_best_right_split, y_best_right_split, parent_node)
            parent_node.child_left, parent_node.child_right = child_node_left, child_node_right
            parent_node.decision = (split_feature, split_threshold_value)
            return child_node_left, child_node_right, True
        else:
            return None, None, False

    def predict_sample(self, X_sample):

        # Starting node is the root node,
        current_node = self.nodes[0]

        # Looping until we reach a terminal node (traversing the tree),
        while current_node.decision is not None:

            # Extracting decision,
            feature_idx, threshold_value = current_node.decision

            # Making the decision,
            if X_sample[feature_idx] <= threshold_value:
                current_node = current_node.child_left
            else:
                current_node = current_node.child_right

        # Returning prediction from the terminal node,
        return current_node.prediction
    
    def score(self, X, y):

        # Storing predictions as an array,
        y_preds = np.array([self.predict_sample(x) for x in X])

        # Computing mean relative error,
        score = np.mean(np.abs((y-y_preds)/y))

        return score
    
    def info(self):
        
        # Printing node info,
        for node in self.nodes:
            node.info(verbose=True)

        return None

class Node():
    """The class for node objects. Essentially used as a container."""

    def __init__(self, X, y, parent_node):
        """Constructor method for the node. Class variables contain node information and encode its location in the tree
        required for predictions."""

        # Node information,
        self.X, self.y = X, y
        self.decision = None
        self.prediction = None

        # Encodes location in the tree,
        self.parent, self.child_left, self.child_right = parent_node, None, None

    def info(self, verbose=False):
        """Returns information about the node."""

        if verbose:
            print(f"Parent: {self.parent}, Decision: {self.decision}, Prediction: {self.prediction}")

        return self.parent, self.decision, self.prediction