import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from keras.callbacks import History 

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

st.title('SmArtCrop: The Crop Recommendation System')

st.success(""" Neural Network Implementation """)

data = pd.read_csv("D:/rimjhim/SEMESTERS/Semester 6 (10th january 2022)/Foundations of Data science/Project/apy1.csv")
st.title("Crop Data")
st.write(data)

data = data[data['State_Name'] == "Andhra Pradesh"]
data['Production'] = pd.to_numeric(data['Production'],errors = 'coerce')
data['State_Name'] = pd.to_numeric(data['State_Name'],errors = 'coerce')
data['District_Name'] = pd.to_numeric(data['District_Name'],errors = 'coerce')
data['District_Name'] = pd.to_numeric(data['District_Name'],errors = 'coerce')


data['Yield'] = data['Production']/data['Area']

C_mat = data.corr()

fig = plt.figure(figsize = (10,8))
sns.heatmap(data.corr(),annot=True, cmap="Greens")
st.pyplot(fig)
data = data[data['Crop_Year']>=2004]

data = data.join(pd.get_dummies(data['District_Name']))
data = data.join(pd.get_dummies(data['Season']))
data = data.join(pd.get_dummies(data['Crop']))
data = data.join(pd.get_dummies(data['Crop_Year']))
data = data.join(pd.get_dummies(data['State_Name']))

data= data.drop('District_Name', axis=1)
data = data.drop('Season',axis=1)
data = data.drop('Crop',axis=1)
data = data.drop('Crop_Year', axis=1)
data = data.drop('Production', axis=1)
data = data.drop('State_Name', axis=1)
st.write('Shape of dataset:', data.shape)
st.write('Number of classes:', len(np.unique(data)))
from sklearn import preprocessing
# Create x, where x the 'scores' column's values as floats
x = data[['Area']].values.astype(float)
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)


data['Area'] = x_scaled


data = data.fillna(data.mean())
b = data['Yield']
a = data.drop('Yield', axis = 1)
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.3, random_state = 42)
a_train = sc.fit_transform(a_train)
a_test = sc.transform(a_test)


NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = a_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

history = History()
History = NN_model.fit(a_train, b_train, epochs=50, batch_size=500, validation_split = 0.2, callbacks=[history])

#st.write(history.history.keys())

st.markdown('**EVALUATION PARAMETERS**')
st.latex(r''' Mean  Absolute  Error(MAE) = \sum_{i=1}^{D}|x_i-y_i| ''')

fig = plt.figure(figsize=(10, 6))
plt.plot(History.history['mean_absolute_error'])
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
plt.plot(History.history['mean_absolute_error'])
plt.plot(History.history['val_mean_absolute_error'])
plt.title('mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
st.pyplot(fig)

#summarize history for loss
fig = plt.figure(figsize=(10, 6))
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
st.pyplot(fig)