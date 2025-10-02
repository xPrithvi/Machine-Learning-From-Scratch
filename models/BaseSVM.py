# Dependencies,
import numpy as np
from cvxopt import matrix, solvers

class BaseSVC:
    """Class for the support vector classifier (SVC)."""

    def __init__(self, C=1, kernel="linear", kernel_params=None):
        """Constructor method."""

        # Model parameters,
        self.weights, self.bias = None, None
        self.label = "support vector classifier (soft-margin)"
        self.kernel, self.kernel_params = kernel, kernel_params
        self.fitted, self.scored = False, False
        self.C = C

        # Training data,
        self.X, self.y = None, None
        self.n_samples, self.n_features = None, None
        self.model_score = None

        # Related to the optimisation algorithm,
        self.Q_matrix = None
        self.alphas = None
        self.sv_idxs = None
        self.n_sv = None

    def fit(self, X, y):
        """Use this method to fit the model."""

        # Assigning data properties,
        self.X, self.y = X, np.where(y <= 0, -1, 1) # <-- Re-labeling class labels.
        self.n_samples, self.n_features = X.shape[0], X.shape[1]

        # Computing our Lagrangian multipliers,
        self.alphas = self._solve_dual_()

        # Computing model parameters,
        self.weights = np.dot(self.alphas*self.y, self.X)
        self.bias = np.mean([self.y[i] - np.dot(self.weights, self.X[i]) for i in self.sv_idxs])

        # Update fitted state,
        self.fitted = True

    def predict(self, X):
        """This method returns the predictions when supplied with samples."""
        return np.sign(self._decision_function(X)).astype(int)

    def score(self, X, y):
        """Computes the classification accuracy on the provided data."""

        # Re-labeling class labels,
        y = np.where(y <= 0, -1, 1)
        
        # Computing predictions,
        y_pred = self.predict(X)

        # Calculating classification accuracy,
        accuracy = np.mean(y_pred == y)
        self.model_score = accuracy
        self.scored = True

        return accuracy

    def _solve_dual_(self, verbose=False, epsilon=1e-5):
        """Finds the Lagrange multipliers which maximise the dual function for the hard-margin SVC."""

        # Constructing the Q matrix (weighted Gram matrix),
        self.Q_matrix = np.outer(self.y, self.y)*np.dot(self.X, self.X.T)
        ones_vector = np.ones(self.n_samples, dtype=np.double)

        """Translating into CVXOPT formalism."""

        # Objective function,
        self.Q_matrix = np.outer(self.y, self.y) * np.dot(self.X, self.X.T)
        P_matrix_obj = matrix(self.Q_matrix.astype(np.double)) # <-- Wrapping the matrix  
        q_vector_obj = matrix(-1*ones_vector)

        # Constraint (1),
        G_std = -np.eye(self.n_samples)
        h_std = np.zeros(self.n_samples)

        G_slack = np.eye(self.n_samples)
        h_slack = np.ones(self.n_samples) * self.C

        G_object = matrix(np.vstack((G_std, G_slack)))
        h_object = matrix(np.hstack((h_std, h_slack)))

        # Constraint (2),
        A_object = matrix(self.y.reshape(1, -1).astype(np.double))
        b_object = matrix([0.0])

        # Solving,
        if not verbose:
            solvers.options['show_progress'] = False
        sol = solvers.qp(P=P_matrix_obj , q=q_vector_obj, G=G_object, h=h_object, A=A_object, b=b_object)
        alphas = np.asarray(sol["x"]).flatten() # <-- Extracting Lagrange multipliers.

        # Extracting support vectors,
        self.sv_idxs = np.where(alphas > epsilon)[0]
        self.n_sv = len(self.sv_idxs)

        return alphas

    def _decision_function(self, X):
        """Returns the distance a sample is from the decision boundary in feature space."""
        return np.dot(X, self.weights) + self.bias
    
    def _compute_slack(self, epsilon_bound=1e-3):
        """Computes the slack variable for each training sample and its assoiated Lagrange multiplier. Returns a tuple."""

        # Computing the lagrange multipliers,
        slack_multipliers = self.C - self.alphas

        # Computing slack variables,
        slack_vars = np.zeros(shape=self.n_samples)
        mask = (self.alphas > self.C - epsilon_bound) & (self.alphas < self.C + epsilon_bound) # <-- We create a mask.
        selected_idxs = mask.nonzero()[0] # <-- Extracting indices where condition was met.
        for idx in selected_idxs:
            slack_vars[idx] = 1 - self.y[idx]*(np.dot(self.X[idx], self.weights) + self.bias)
        
        return slack_vars, slack_multipliers
    
    def _repr_html_(self):
        """Compact HTML GUI as the object representation in Jupyter Notebook."""
        html = f"""
        <div style="
            border:1px solid black;
            border-radius:6px;
            font-family:Arial, sans-serif;
            font-size:12px;
            line-height:1.2;
            width:fit-content;
            background:white;
            color:black;
            padding-left:8px;
            padding-right:8px;
        ">
            <!-- Title bar -->
            <i>{self.label}</i>
            <div style="
                background:#e0e0e0;
                padding:3px 6px;
                font-weight:bold;
                border-bottom:1px solid black;
                border-top-left-radius:6px;
                border-top-right-radius:6px;
                color:black;
            ">
                SVC
                <div style="margin-top:2px;">
                    <img src="svc_icon.png" alt="tree icon" width="30" height="30">
                </div>
            </div>

            <!-- Hyperparameters -->
            <ul style="margin:4px 0 4px 16px; padding:0;">
                <b>Hyperparameters:</b><br>
                self.C:</b> {self.C}<br>
                self.kernel:</b> {self.kernel}<br>
                self.kernel_params:</b> {self.kernel_params}<br>
            </ul>

            <!-- Divider -->
            <div style="
                border-top:1px solid #ccc;
                margin:4px 0;
            "></div>

            <!-- Status and other info -->
            <ul style="margin:4px 0 4px 16px; padding:0;">
        """

        if self.fitted:
            html += "<b>Status:</b> <span style='color:green;'>Fitted</span><br>"
            html += f"Score:</b>{round(self.model_score, 3) if self.scored == True else None}<br>"
            html += f"self.n_features:</b> {self.n_features}<br>"
            html += f"self.n_samples:</b> {self.n_samples}<br>"
            html += f"self.n_sv:</b> {self.n_sv}<br>"
        else:
            html += "<b>Status:</b> <span style='color:red;'>Not Fitted</span><br>"

        return html