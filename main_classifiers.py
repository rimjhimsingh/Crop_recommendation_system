import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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
st.success(""" Explore different classifier and datasets Compare the performace of each model """)

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('crop_recommendation', '')
)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('SVM', 'Random Forest', 'Naive Bayes','KNN')
)


#displaying the dataset
def get_dataset(name):
    data = None
    if name == 'crop_recommendation':
        
        data = pd.read_csv("D:/rimjhim/SEMESTERS/Semester 6 (10th january 2022)/Foundations of Data science/Project/Intelligent_CropPrediction_System-main/Data-processed/crop_recommendation.csv")
        st.title("Crop Data")
        st.write(data)
        X = data
        y = data.label
    return X, y


st.markdown('**EVALUATION PARAMETERS**')
st.latex(r''' Accuracy = \frac{TP+TN}{TP+TN+FP+FN} ''')
st.latex(r''' Precision = \frac{TP}{TP+FP} ''')
st.latex(r''' Recall = \frac{TP}{TP+FN} ''')
st.latex(r''' F1 score = \frac{2*precision * recall}{precision + recall} ''')

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))



def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 1.0)
        params['C'] = C

    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K

    elif clf_name == 'Decision Tree':
        depth = st.sidebar.slider('Depth', 1, 5)
        st.write(Accuracy = "90.68181818181819")
        params['Depth'] = depth

    elif clf_name == 'Naive Bayes':
        params[''] = None

    else: #random forest
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 200)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])

    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])

    elif clf_name == 'Decision Tree': 
        clf = DecisionTreeClassifier(max_depth = params['Depth'], random_state = 2)
        st.write(Accuracy = "90.68181818181819")

    elif clf_name == 'Naive Bayes':
        NaiveBayes = GaussianNB()
        clf = GaussianNB()

    #elif clf_name == 'Neural Network':
      # clf= NeuralNetwork()
    
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=2)

    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

features = X[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = X['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
clf.fit(Xtrain, Ytrain)
y_pred = clf.predict(Xtest)

#print accuracy score
acc = accuracy_score(Ytest, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)



#### PLOT DATASET ####
fig = plt.figure()
sns.lineplot( X['temperature'], X['rainfall'])
st.pyplot(fig)

fig = plt.figure(figsize=(10, 4))
sns.countplot(x = "label", data = X)
st.pyplot(fig)

fig = plt.figure(figsize=(10, 4))
sns.scatterplot(X['temperature'], X['rainfall'])
st.pyplot(fig)

fig = plt.figure(figsize=(10, 4))
sns.distplot( X['ph'])
st.pyplot(fig)

#fig = plt.figure(figsize=(10, 10))
#sns.pairplot(X)
#st.pyplot(fig)

fig = plt.figure(figsize=(10, 4))
sns.heatmap(X.corr(),annot=True)
st.pyplot(fig)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Temprature', 'humidity', 'ph'])
st.area_chart(chart_data)
st.bar_chart(chart_data)