import numpy as np
import pandas as pd   
import pickle




df = pd.read_csv("balone.data", names = ['Sex', 'Length','diameter', 'height', 'Whole weight', 'shucked weight', 'Viscera weight', 'Shell weight', 'rings'] )
df = pd.get_dummies(df)
df = df[['Length','diameter', 'height','Whole weight','shucked weight','Viscera weight','Shell weight','Sex_F','Sex_I','Sex_M','rings']]
df = df.rename(columns = {'rings' : 'age'})
X = df.iloc[:, 0:-1].values
y= df.iloc[: , -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
from sklearn.neighbors import KNeighborsClassifier

knn =KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p=2)
knn.fit(X_train,y_train)


pickle.dump(knn, open('model.pkl', 'wb'))

mod = pickle.load(open('model.pkl', 'rb'))
print(mod.predict([[0.454,0.342, 0.0564, 0.434,0.2231,0.16,0.213,0,0,1]]))
    

        
