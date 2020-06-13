import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense,Dropout

def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


dataset = pd.read_csv("Churn_Modelling.csv")

print(dataset.info())
print(dataset.describe())

X = dataset[
    ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
     "EstimatedSalary"]]
y = dataset["Exited"].values

# Encoding categorical data
# Encoding the Independent Variable

# RowNumber,CustomerId,Surname,
numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                    "EstimatedSalary"]
categorical_features = ["Geography", "Gender", ]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='error', drop="first"))
])
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(missing_values=np.nan)),
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='passthrough'
)
X = preprocessor.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)


mean = accuracies.mean()
variance = accuracies.std()