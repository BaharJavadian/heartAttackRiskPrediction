
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Applying and Checking Accuracy and Precision of different Classification Models
models = {'Logistic_Regression':LogisticRegression(),
          'Random_Forest':RandomForestClassifier(),
          'Decision_Tree':DecisionTreeClassifier(random_state=42),
          'SVM':SVC(kernel = 'rbf'), 
          'KNN':KNeighborsClassifier(n_neighbors = 10),
          'Naive_Bayes':GaussianNB()}


    
def prediction_models(X, y, test_x, test_y):
    """
    Calculate accuracy, percision and confusion Matix for different models
    params X: train features data 
    params y: train target data
    params test_x: test feature data
    params test_y: test target data
    
    """
    for i in models: 
        obj = models[i]
        print(obj)
        obj.fit(X, y)
        obj_pred = obj.predict(test_x)
        accuracy = accuracy_score(test_y,obj_pred)
        precision = precision_score(test_y,obj_pred,zero_division=1)
        print('Accuracy of '+i+': ',accuracy)
        print('Precision of '+i+': ',precision)
        print(f'Classification Report:\n{classification_report(test_y, obj_pred)}')
        print(f'Confusion Matrix:\n{confusion_matrix(test_y, obj_pred)}')

def boxplot_outliers(df, columns_to_visualize):
    # Columns to visualize
    columns_to_visualize = ['Age','Cholesterol',
       'Heart Rate', 'Income', 'BMI']

    # Plot each specified column
    for column in columns_to_visualize:
        plt.figure(figsize=(5, 2.5))
        sns.boxplot(x=column, data=df)  # Set x as column name and specify the data argument
        plt.title(f'Boxplot of {column}')
        plt.show()
  
