import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.models
import tensorflow as tf
import tensorflow.python.keras
import get_feature as gt
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
print("start_to_load_tests")
df=pd.read_csv('./Data/test_data.csv')
df=df.iloc[0:,1:]
'''
df = gt.extract_audio_features('./Data/test')
df.to_csv('./Data/test_data.csv', index=False)'''


print("done")
label_index={"<DirEntry 'blues": 0, "<DirEntry 'classical": 1, "<DirEntry 'country": 2, "<DirEntry 'disco": 3, "<DirEntry 'hiphop": 4, "<DirEntry 'jazz": 5, "<DirEntry 'metal": 6, "<DirEntry 'pop": 7, "<DirEntry 'reggae": 8, "<DirEntry 'rock": 9}
index_label={0: "<DirEntry 'blues", 1: "<DirEntry 'classical", 2: "<DirEntry 'country", 3: "<DirEntry 'disco", 4: "<DirEntry 'hiphop", 5: "<DirEntry 'jazz", 6: "<DirEntry 'metal", 7: "<DirEntry 'pop", 8: "<DirEntry 'reggae", 9: "<DirEntry 'rock"}
df_X=df.loc[:,df.columns!='label']
df_X=df_X.to_numpy()


df_y=df['label']
df_y=df_y.map(label_index)
df_y=df_y.to_numpy()
#print(df_y)

scaler=joblib.load('scaler.save')

df_X=scaler.transform(df_X)



model=keras.models.load_model("my_model")
preds = model.predict(df_X)

ans=[]

for i in range(len(preds)//10):
    pr=preds[i*10:i*10+10]
    all=np.zeros(len(pr[0]))
    for j in range(len(pr)):
        all+=pr[j]
    maxx=-1
    tag=-1
    for j in range(len(all)):
        if all[j]>maxx:
            maxx=all[j]
            tag=j
    ans.append(tag)


a=np.argmax(preds,axis=1)
print(len(a))

acur=0
for i in range(len(a)):
    if(a[i]==df_y[i]):
        acur+=1

acur/=len(a)
print(round(accuracy_score(df_y, a),5))

