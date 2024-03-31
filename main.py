import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# Load datasets
df1 = pd.read_csv('./Datasets/data.csv',sep=";|,")
df = df1  # Use a single dataset for simplicity


#df.columns = df.columns.str.strip()
# Define the features and target variable
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']
target = 'cardio'

# Extract features and target variable
X = df[features]
y = df[target]




# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'active'])

# Handle missing values in features using imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Handle missing values in the target variable
imputer_y = SimpleImputer(strategy='mean')
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1))  # Reshape for 2D array

# Convert back to a Pandas Series
y_imputed = pd.Series(y_imputed.flatten(), name=target)

# Convert target variable to binary format
threshold = 0.5  # You can adjust this threshold based on your needs
y_binary = (y_imputed > threshold).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_binary, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create ensemble
class StackedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_models_ = [list() for x in base_models]

    def fit(self, X, y):
        # Split the training data to train base and meta models
        X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.5, random_state=42)
        
        meta_features = np.zeros((X_meta.shape[0], len(self.base_models)))
        
        # Train each base model and predict on the meta data to create new features for the meta model
        for i, model in enumerate(self.base_models):
            instance = clone(model)
            instance.fit(X_base, y_base)
            self.base_models_[i].append(instance)
            meta_features[:, i] = instance.predict(X_meta)
        
        # Train the meta model on the new feature set created from base models' predictions
        self.meta_model.fit(meta_features, y_meta)
        return self

    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # Predict with all base models and use these predictions as features for the meta model
        for i, models in enumerate(self.base_models_):
            predictions = np.column_stack([model.predict(X) for model in models])
            meta_features[:, i] = predictions.mean(axis=1)
        
        return self.meta_model.predict(meta_features)

# Create models
logistic_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
nb_model = GaussianNB() 
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

stacked_ensemble1 = StackedEnsemble(base_models=[xgb_model, svm_model], meta_model=logistic_model)
stacked_ensemble2 = StackedEnsemble(base_models=[logistic_model, svm_model], meta_model=xgb_model)
stacked_ensemble3 = StackedEnsemble(base_models=[xgb_model, logistic_model], meta_model=svm_model)

# Train models with progress bar
for model in tqdm([logistic_model, rf_model, svm_model, nb_model, knn_model, dt_model, xgb_model, stacked_ensemble1, stacked_ensemble2, stacked_ensemble3], desc="Training Models"):
    model.fit(X_train, y_train)

# Make predictions on the test set with progress bar
y_preds = [model.predict(X_test) for model in tqdm([logistic_model, rf_model, svm_model, nb_model, knn_model, dt_model, xgb_model, stacked_ensemble1, stacked_ensemble2, stacked_ensemble3], desc="Making Predictions")]

# Evaluate models and choose the best
best_model = None
best_accuracy = 0

for i, (model, y_pred) in enumerate(zip([logistic_model, rf_model, svm_model, nb_model, knn_model, dt_model, xgb_model, stacked_ensemble1, stacked_ensemble2, stacked_ensemble3], y_preds)):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Model {i + 1} - {type(model).__name__}:")
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}')
    print('\n')

    # Choose the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"The best model is {type(best_model).__name__} with an accuracy of {best_accuracy}")
