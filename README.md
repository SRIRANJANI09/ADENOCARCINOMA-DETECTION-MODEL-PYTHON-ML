# ADENOCARCINOMA-DETECTION-MODEL-PYTHON-ML

__Importing Required Libraries__
load_breast_cancer: This function loads the built-in breast cancer dataset from Scikit-learn.
train_test_split: Splits the data into training and testing sets.
GaussianNB: Implements the Naive Bayes classifier.
accuracy_score: Calculates the accuracy of the classifier.

__Loading the Dataset__
The breast cancer dataset is loaded into the variable data, which is a dictionary-like object containing:
data["data"]: Feature values of the dataset.
data["target"]: Class labels (malignant or benign).
data["feature_names"]: Names of the features (e.g., 'mean radius', 'mean texture', etc.).
data["target_names"]: The target label names (malignant or benign).

__Extracting Features and Labels__
label_names: Stores the class names ('malignant', 'benign').
labels: Stores the class labels as integers (0 for malignant and 1 for benign).
feature_names: Stores the names of the features.
features: Stores the feature values (measurements for each tumor).

__Viewing the Data__
This prints:
label_names: Displays the possible classifications: 'malignant' and 'benign'.
labels[0]: The label of the first data instance (0 means malignant).
feature_names: The names of all features used to describe the tumors.
features[0]: The actual feature values for the first tumor in the dataset.

__Splitting the Dataset__
train_test_split(): Splits the data into a training set and a test set.
train: Features used to train the model (80% of the data).
test: Features used for testing the model (20% of the data).
train_labels: Labels for training data.
test_labels: Labels for testing data.
test_size=0.2: Reserves 20% of the dataset for testing.
random_state=42: Sets a seed for reproducibility.

 __Training the Naive Bayes Model__
 GaussianNB(): Initializes the Naive Bayes classifier.
gnb.fit(train, train_labels): Trains the classifier using the training data and corresponding labels.
__Making Predictions__
gnb.predict(test): The trained model predicts the labels for the test set.
preds: Stores the predicted labels (0 or 1) for the test data, and the predictions are printed.

__Evaluating Model Performance__
accuracy_score(test_labels, preds): Calculates the accuracy by comparing the true labels (test_labels) with the predicted labels (preds).
The accuracy of the model is 97.36%, meaning the model is correct 97% of the time.
__How the Naive Bayes Model Works__
Naive Bayes is a probabilistic classifier based on Bayes' Theorem with an assumption of independence between the features.

𝑃
(
𝐶
∣
𝑋
)
=
𝑃
(
𝑋
∣
𝐶
)
⋅
𝑃
(
𝐶
)
𝑃
(
𝑋
)
P(C∣X)= 
P(X)
P(X∣C)⋅P(C)
​
 
P(C|X): Probability of class C given features X.
P(X|C): Likelihood of features X given class C.
P(C): Prior probability of class C.
P(X): Probability of the features X.
The Gaussian Naive Bayes assumes that the features follow a normal (Gaussian) distribution.

__Key Points:__
Independence Assumption: Naive Bayes assumes that all features are independent of each other, which simplifies calculations but may not always be true in real-world datasets.
Efficiency: Naive Bayes is a fast and simple algorithm, making it ideal for large datasets.
Good for binary classification: Since the dataset has two target classes (malignant/benign), Naive Bayes is well-suited for this task.
