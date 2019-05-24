import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_excel('Geostat Timah.xlsx',sheet_name='Assay')

#preparing data for training and testing
Afull = df[['Thick','Sn (kg/m3)']].values
A = df[['Thick','Sn (kg/m3)']].head(12).values
B = ['OB','OB','Miencan','Miencan','Kong','Kong','Kaksa','Kaksa','Kaksa','Kaksa','Kong','Kong']

#initializing functions for weight average
def my_agg(x):
    names = np.average(x['Sn (kg/m3)'],weights=x['Thick'],axis=0)
    return pd.Series(names,index=['wavg'])

def my_thick(x):
    tot_thick = np.sum(x['Thick'],axis=0)
    return pd.Series(tot_thick,index=['Thick Sumz'])

#model fitting
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(A,B)

#model testing
prediction_tree = clf_tree.predict(A)
acc_tree = accuracy_score(B,prediction_tree)*100
print('Accuracy for Tree: {}'.format(acc_tree))
print(prediction_tree)

#utilizing model with full data
prediction_print = clf_tree.predict(Afull)
df['resultz'] = prediction_print

#executing weight averages function and from to function
df1 = df.groupby(['BHID','resultz'],sort=False).apply(my_agg)
df2 = df.groupby(['BHID','resultz'],sort=False).apply(my_thick)
df3 = df.groupby(['BHID','resultz'],sort=False)['From'].first()
df4 = df.groupby(['BHID','resultz'],sort=False)['To'].last()

#exlusively picking weight average columns
wavg = df1['wavg']
total_thick = df2['Thick Sumz']

#combining results
fin = pd.concat([wavg,total_thick,df3,df4], axis=1)

#exporting to excel and printing it out
#fin.to_excel('hasil final.xlsx')
print(fin)
