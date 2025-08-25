# Machine Learning From Scratch
The most common machine learning models and algorithms programmed from first principles. The aim is to implement these models and algorithms to the same standard as widely used libraries such as scikit-learn with clean, efficient code, clear modular design, and rigorous adherence to the underlying mathematics.

| Model / Algorithm            | Type                      | Completed? | Notes |
|------------------------------|---------------------------|----------|------------|
| Linear Regression            | Regression                | ✅        | TBR        |
| Logistic Regression          | Classification            | ✅        | TBR        |
| K-Nearest Neighbours         | Classification            | ✅        | TBR        |
| K-Means                      | Clustering                | ✅        | TBR        |
| Stochastic Gradient Descent  | Optimisation              | ✅        | TBR        |
| Naive Bayes Classifier       | Classification            | Categorical ✅, Multinomial ❌, Gaussian ❌ | TBR        |
| Decision Tree                | Classification/Regression | Classifier ✅, Regressor ✅, CCP Pruning ✅, Feature Importance ✅ | Speed and efficiently need to be optimised for usage in random forests. |
| Random Forest                | Classification/Regression | Bootstrap aggregated ✅, Rotation forest ❌, Extremely Randomised Trees (ERT) ❌ | TBR        |
| Support Vector Machine       | Classification/Regression | ⏳🚧      | Old code exists. Needs to be reviewed and rewritten to a higher standard. |
| Principal Component Analysis | Dimensionality Reduction  | ✅        | First principle derivation of the eigenvalue equation need to be added. |
| DBSCAN                       | Clustering                | ✅        | Pseudo-code needs to be added for queuing algorithm used for cluster growth. |
| Gaussian Mixture Models      | Clustering (Probabilistic)| ⏳🚧      | In progress (put on hold). Need to formally explore Maximum Likelihood Estimation (MLE) theory. |
| Linear Discriminant Analysis | Dimensionality Reduction  | ❌        | Not started.    |
| Gradient Boosting            | Classification/Regression | ⏳🚧      | Basic algorithm written. Theory section needs to be fleshed out more. Decision Tree Regressor must be created before creating boosted forests.|
