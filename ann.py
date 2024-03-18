import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data_sheet.csv")
df.head()
df.sample(5)
df.shapes
df.isnull().sum()
df.duplicated().sum()

from skimpy import skim
skim(df)
df.info()

num_cols = df.select_dtypes(include=["float64", "int64"]).columns
num_data = df[num_cols]
num_data.corr()["Exited"]

df["Geography"].value_counts().plot(kind='pie', autopct="%.2f")
df["Gender"].value_counts().plot(kind="pie", autopct="%.2f")

plt.figure(figsize=(20, 10))
plot_no = 1
for col in num_data:
    plt.subplot(5, 5, plot_no)
    sns.boxplot(x=df[col])
    plot_no += 1
plt.tight_layout()

cols_to_plot = num_data.columns
num_cols = len(num_data.columns)
num_rows = num_cols // 5 + 1 if num_cols % 5 != 0 else num_cols // 5

fig, axes = plt.subplots(nrows=num_rows, ncols=5, figsize=(20, 4 * num_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    sns.histplot(df[col], ax=ax, kde=True)
    ax.set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

sns.pairplot(df)
df.columns

df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
df.head()

x = df.drop(columns=["Exited"])
y = df["Exited"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.fit_transform(X_test)

X_test_scaled
X_train_scaled

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(13, activation="sigmoid", input_dim=13))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

hist = model.fit(X_train, y_train, epochs=10)
hist.history

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()

model.layers[0].get_weights()
model.predict(X_test_scaled)

y_pred = np.where(model.predict(X_test_scaled) > 0.5, 1, 0)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

model1 = Sequential()
model1.add(Dense(13, activation="relu", input_dim=13))
model1.add(Dense(13, activation="relu"))
model1.add(Dense(1, activation="sigmoid"))
model1.summary()

model1.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
hist1 = model1.fit(X_train_scaled, y_train, epochs=40, validation_split=0.2)
hist1.history

import matplotlib.pyplot as plt
plt.plot(hist1.history['loss'])
plt.plot(hist1.history["val_loss"])
plt.show()

model1.layers[0].get_weights()
model1.predict(X_test_scaled)

y_pred = np.where(model1.predict(X_test_scaled) > 0.5, 1, 0)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist1.history["loss"])
plt.show()