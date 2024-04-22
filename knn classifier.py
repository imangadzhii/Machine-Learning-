from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

knn_model = KNeighborsClassifier()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2\5)

knn_model.fit(x_train,y_train)

y_pred = knn_model.predict(x_test)

acc = accuracy_score(y_test,y_pred)

print(f'accuracy:{acc*100}')
