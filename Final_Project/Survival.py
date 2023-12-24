# Inputs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns

# Functions

def remove_outliers(dataframe, column_name, multiplier=1.5):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df_no_outliers = dataframe\
        [(dataframe[column_name] >=lower_bound) &\
         (dataframe[column_name] <= upper_bound)]
    
    return df_no_outliers


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.show()

# Data preparation

scaler = StandardScaler()

df = pd.read_csv('Survival_dataset.csv')

df = df.drop(columns=['Survival', 'recordid'])

df = df.fillna(0)

# Without removing: 0.8510869741439819
# With removing: 0.8833718299865723
df = remove_outliers(df, 'Length_of_stay')

X = df.drop('In-hospital_death', axis=1)
y = df['In-hospital_death']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Experements

df.corrwith(df['In-hospital_death']) # looking at Length_of_stay
# 0.033884 is not bad correlation

# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.regplot(x="Length_of_stay", y="In-hospital_death",\
#             data=df, logistic=True, scatter_kws={'s': 50}, ci=None)
# plt.xlabel('Length of Stay')
# plt.ylabel('In-hospital Death')
# plt.title('Relationship between In-hospital Death and Length of Stay')
# plt.show()

# The longer people stay in hospital, 
# the more likely that they will die.

# NN Model

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
    
model.compile(optimizer='adam', loss='binary_crossentropy',\
              metrics=['accuracy'])

epochs = 9

history = model.fit(X_train_scaled, y_train, epochs=epochs,\
                    batch_size=32, validation_split=0.2)
plot_history(history)

accuracy = model.evaluate(X_test_scaled, y_test)[1]
print(f"Test Accuracy: {accuracy}")

# Testing on unseen data

unseen_X = pd.read_csv('unseen_survival.csv')

unseen_X = unseen_X.drop(['recordid', 'Survival',\
                          'In-hospital_death'], axis=1)

unseen_X = unseen_X.fillna(0)

unseen_X_scaled = scaler.transform(unseen_X)

predictions = model.predict(unseen_X_scaled)

# Prediction on unseen data result

threshold = 0.5 

binary_predictions = (predictions > threshold).astype(int)

if binary_predictions[0] == 1:
    print('The patient will die in the hospital.')
else:
    print('The patient will survive in the hospital.')
