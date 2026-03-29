#Introduction
#Exploratory Data Analysis
#Data Preprocessing
#Main Problem Of The Dataset
#Classic Algorithms Of ML
#Isolation Forest
#AutoEncoders 1 - Deep NN Architecture
#AutoEncoders 2 - Deeper NN Architecture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data_= pd.read_csv('/Users/srirakeshnagasai/Downloads/project-1/data_train.csv')
data_

data_.head()

print("Shape:", data_.shape, '\n')
print("Data Types:", data_.dtypes, '\n')
for col in data_.columns:
    print(f"Unique Data Of The {col}:", data_[col].unique())

print('Nan Type Values:\n', data_.isnull().sum())


missing_cols = [col for col in data_.columns if data_[col].isnull().sum() > 0]

fig, axes = plt.subplots(1, len(missing_cols), figsize=(5 * len(missing_cols), 5))

if len(missing_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, missing_cols):
    sns.histplot(data_[col], kde=True, ax=ax, color='red')
    ax.set_title(col)

plt.figure(figsize=(10, 10))
plt.tight_layout()
plt.show()

for col in data_.columns:
    if data_[col].dtype == 'object':  
        data_[col] = data_[col].fillna('Unknown')  
    else:  
        data_[col] = data_[col].fillna(data_[col].mean())  

data_.isnull().sum()

print("Unique Data of Transaction_ID:", len(data_['Transaction_ID'].unique()))
print("Unique Data Of User_ID:", len(data_['User_ID'].unique()))

data_.drop(['Transaction_ID'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data__eda = data_.copy()
data__train = data_.copy()

categorical_features = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']

for col in categorical_features:
    data__eda[col] = le.fit_transform(data__eda[col])

data__train = pd.get_dummies(data__train, columns=categorical_features)

print('Train Dataset:')
data__train.head()

print('EDA Dataset:')
data__eda.head()

plt.figure(figsize=(12,12))

sns.heatmap(data__eda.corr(), annot=True)

sns.countplot(x=data_['Fraudulent']) 

from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(data__eda.drop(['Fraudulent'], axis=1), data__eda['Fraudulent'])
print(mi_scores)

important_features = data__eda.drop(['Fraudulent'], axis=1).columns[mi_scores > 0]
important_features

fig, axes = plt.subplots(nrows=1, ncols=len(important_features), figsize=(20, 5))

for ax, col in zip(axes, important_features):
    sns.boxplot(x=data__eda[col], ax=ax)

plt.tight_layout()
plt.show()

##Modelling with Classic Machine Learning Algorithms

data__train.shape

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

data_train = pd.read_csv("data_train.csv")  
X = data__train.drop(['Fraudulent'], axis=1)
Y = data__train['Fraudulent']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
from tqdm import tqdm

scaler = StandardScaler()

X = data__train.drop(['Fraudulent'], axis=1)
Y = data__train['Fraudulent']

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X, Y)
X_scaled = scaler.fit_transform(X_train_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled)

models = {
    "Logistic Regression": LogisticRegression(C=1.0, solver="liblinear", max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight="balanced_subsample", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10, criterion="gini", class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, num_leaves=60, min_data_in_leaf=5, force_col_wise=True, scale_pos_weight=5, verbose=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, min_samples_split=5),
    "SGD Classifier": SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, max_iter=2000, tol=1e-4, class_weight="balanced"),
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.03, depth=7, l2_leaf_reg=5, scale_pos_weight=5, verbose=0),
}

results = []
for name, model in tqdm(models.items(), desc="Training Models", total=len(models)):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_pred)

    results.append({"Model": name, "Accuracy": accuracy, "F1 Score": f1, "ROC AUC": roc_auc})

    plot_confusion_matrix(y_test, y_pred, name)

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print(results_df)

results_df


from sklearn.ensemble import IsolationForest

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)

y_pred_train = iso_forest.predict(X_train)
y_pred_test = iso_forest.predict(X_test)

y_pred_test = [1 if x == -1 else 0 for x in y_pred_test]

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix - Isolation Forest")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU not found. Using CPU.')
else:
    print(f'GPU found: {device_name}')

X = df_train.drop(['Fraudulent'], axis=1)
Y = df_train['Fraudulent']
X, Y = smote.fit_resample(X, Y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42, stratify=Y)

model = Sequential([
    Dense(512, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(256, activation='relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),

  Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])


model = Sequential([
    Dense(1024, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(256, activation='relu'),
    Dropout(0.4),
    BatchNormalization(),

    Dense(128, activation='relu'),
    Dropout(0.4),
    BatchNormalization(),

    Dense(64, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),

    Dense(16, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),

    Dense(8, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),

    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
