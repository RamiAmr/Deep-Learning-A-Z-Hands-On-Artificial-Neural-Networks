import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_classifier(optimizer="adam", n_layers=2, layer_units=6, dropout_p=0.1):
    print(
        "build_classifier(optimizer='{}', n_layers='{}', layer_units='{}', dropout_p='{}')".format(optimizer, n_layers,
                                                                                                   layer_units,
                                                                                                   dropout_p))
    nn = Sequential()

    for i in range(n_layers):
        nn.add(Dense(units=layer_units, kernel_initializer='uniform', activation='relu'))
        nn.add(Dropout(rate=dropout_p, seed=0))

    nn.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    nn.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return nn


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

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {
    "batch_size": [25, 32],
    "epochs": [10, 100, 500],
    "n_layers": [1, 3, 5],
    "layer_units": [6, 8, 10, 12],
    "dropout_p": [0.1, 0.2, 0.5, 0.7],
    "optimizer": ["adam", "rmsprop"]
}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Params {}, Best Score {}".format(best_params, best_score))
