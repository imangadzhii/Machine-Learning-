from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
#iris dataset
iris = load_iris()
x = iris.data
y = iris.target
#train data and test data splittting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)
#assigning model variable
lr = LinearRegression()
#fitting train data into model and training
lr.fit(x_train,y_train)
#predicitng using our model
y_pred = lr.predict(x_test)
acc = mean_squared_error(y_test,y_pred)
#printing accuracy
print(f"accuracy:{acc*100}")
#plotting
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
