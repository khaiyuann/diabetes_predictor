# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:12:30 2022

@author: LeongKY
"""

#%% Imports
import os
import pickle
import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model

#%% Statics
DATA_PATH = os.path.join(os.getcwd(), 'diabetes.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(), 'logs')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
ENC_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
                         
#%% 1. Data loading
df = pd.read_csv(DATA_PATH)
dfc = df.copy()

#%% 2. Data inspection
print(df.describe())
print(df.info())
print(df.head())

#%% 3. Data cleaning
# drop duplicated rows
dfc = dfc.drop_duplicates()
print(dfc.duplicated().value_counts())

# columns where 0 instead of data instead of NaN
cols = ['Glucose', 'SkinThickness', 'BMI', 'Insulin']
dfc[cols] = dfc[cols].replace({0:np.nan})

#observed that high amounts of NaN SkinThickness,Insulin perform KNN imputation
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
dfc['SkinThickness'] = pd.DataFrame(imputer.fit_transform
                                    (np.expand_dims
                                     (dfc['SkinThickness'], -1)))
dfc['Insulin'] = pd.DataFrame(imputer.fit_transform
                              (np.expand_dims
                               (dfc['Insulin'], -1)))

# drop remaining rows containing NaN values
dfc = dfc.dropna(axis=0, how='any')
dfc.plot(kind='box')
plt.show()

#%% 4. Feature selection
# correlation heatmap to determine correlation between features and labels
correlation = dfc.corr()
sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds)
plt.show()

#remove lowest 2 correlation and label columns
drop_cols = ['SkinThickness', 'BloodPressure', 'Outcome']
X = dfc.drop(dfc[drop_cols], axis=1) #feature
y = dfc['Outcome'] #label

#%% 5. Data preprocessing

# standard scaling on selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
pickle.dump(scaler, open(SCALER_PATH, 'wb'))

# encoding labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(np.expand_dims(y,-1))
pickle.dump(encoder, open(ENC_PATH, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.25,
                                                    random_state=13)

#%% 6. Create model
model = Sequential()
model.add(Dense(256, activation='tanh', input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))
model.summary()

# callbacks
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

es_callback = EarlyStopping(monitor='loss', patience=7)
tb_callback = TensorBoard(log_dir=log_files)
callbacks = [es_callback, tb_callback]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')
plot_model(model, os.path.join(os.getcwd(), 'saved_model', 'model.png'))

# train model
hist= model.fit(X_train, y_train, epochs=100,
                 validation_data=(X_test, y_test),
                 callbacks=callbacks)

#%% 7. Evaluate model
y_pred_adv = np.empty([len(X_test),2])
for index, test in enumerate(X_test):
   y_pred_adv[index,:] = model.predict(np.expand_dims(test, 0))

#%% model scoring
y_pred_res = np.argmax(y_pred_adv, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_res))
print(confusion_matrix(y_true, y_pred_res))
print(accuracy_score(y_true, y_pred_res))

#%% model deployment
model.save(MODEL_SAVE_PATH)