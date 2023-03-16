import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import sklearn.preprocessing as skp
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib

df=pd.read_csv('./Data/genre.csv')
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)



df=df.iloc[0:,1:]

y=df['label']
X=df.loc[:,df.columns!='label']

cols=X.columns
scaler = skp.StandardScaler()
df_X = X.to_numpy()
#df_X=pd.DataFrame(df_X,columns=cols)

df_y=y.map(label_index)
df_y=df_y.to_numpy()
X_train=np.array([itm for i,itm in enumerate(df_X) if i%95<80])
y_train=np.array([itm for i,itm in enumerate(df_y) if i%95<80])

X_test=np.array([itm for i,itm in enumerate(df_X) if i%95>=80])
y_test=np.array([itm for i,itm in enumerate(df_y) if i%95>=80])

X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

joblib.dump(scaler,'scaler.save')


'''
for i,x in enumerate(dw.iloc):
    if(i%100<80):
        #print(type(dw.iloc[i]))
        #print(type(dw.iloc[i].to_frame().T))
        df1 = pd.concat([df1 , dw.iloc[i].to_frame().T],ignore_index=True)
    else:
        df2 = pd.concat([df2, dw.iloc[i].to_frame().T], ignore_index=True)

print(len(df1))
print(len(df2))
print(len(df1)/len(df2))'''



#seed=256
#X_train,X_test,y_train,y_test=train_test_split(df_X,df_y,train_size=0.8,random_state=seed,stratify=df_y)

print(y_train)
#print(y_train.to_numpy())


model_2 = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),

    Dense(256, activation='relu'),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(10, activation='softmax'),
])
model_2.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
history = model_2.fit(X_train,
                        y_train,
                        batch_size=128,
                        validation_data=(X_test, y_test),
                        epochs=50)

print(model_2.summary())


model_2.save("my_model")


def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')



'''rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(rforest, "Random Forest")

knn = KNeighborsClassifier(n_neighbors=19)
model_assess(knn, "KNN")'''''

'''nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(256, 128,64,10), random_state=1,batch_size=15,max_iter=50)
model_assess(nn, "Neural Nets")'''
#print(knn.summary())