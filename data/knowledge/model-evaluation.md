# Model evaluation

A held-out test set measures how a fitted model performs on examples it did not see during training. Cross-validation repeats the train-validation process across several partitions and exposes unstable performance that one split can hide.

Evaluation should match the decision. Classification systems may need precision and recall, regression systems often use MAE and R-squared, and clustering systems can use silhouette scores alongside human inspection of the resulting groups.
