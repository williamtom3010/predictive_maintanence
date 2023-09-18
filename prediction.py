import pandas as pd
data = pd.read_excel('/content/predictive_maintenance.xlsx')
print(data)
x = data.iloc[1:,1:6]
print(x)
#before performing oversampling
# from imblearn.over_sampling import ADASYN
# from collections import Counter
# counter = Counter(y)
# print("Before",counter)
#after performing oversampling
# ada = ADASYN(random_state = 130)
# x_new_ada,y_new_ada = ada.fit_resample(x,y)
# counter = Counter(y_new_ada)
# print("After",counter)
# Calculate Pearson correlation matrix
correlation_matrix = x.corr()
# x.corr()
# Print the correlation matrix
print(correlation_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()
y=data.iloc[1:,6]
print(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.7)
from sklearn.svm import SVC
model = SVC()
model=model.fit(xtrain,ytrain)
op=model.predict(xtest)
print(op)
from sklearn.metrics import classification_report
cr=classification_report(ytest,op)
print(cr)
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm=confusion_matrix(ytest,op)
print(cm)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot()



