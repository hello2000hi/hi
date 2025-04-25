# 3. Create a pandas DataFrame and drop rows with missing values
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', None],
    'Age': [25, None, 30]
})
df_clean = df.dropna()
print(df_clean)
# 4. Replace null values in a column with mean of that column
df['Age'] = df['Age'].fillna(df['Age'].mean())
# 2. Add column age_group based on age
def age_group(age):
    if age <= 18:
        return 'Child'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'
df['age_group'] = df['age'].apply(age_group)
# 3. Use train_test_split() to split data
from sklearn.model_selection import train_test_split
X = df[['age', 'sex', 'bmi']]
y = df['has_disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 4. Build a Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
# 5. Plot ROC curve
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
# 6. Calculate confusion matrix and print accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# 4. Build a KNN model to predict loan_status
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
# 5. Use matplotlib to visualize prediction outcomes
import matplotlib.pyplot as plt
y_pred = model.predict(X_test)
plt.scatter(range(len(y_test)), y_test, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', marker='x')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# 4. Build SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# 5. Predict and plot outcomes
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, color='orange')
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='purple')
# 4. Build a Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# 5. Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(y_test, predictions)
# 6. Calculate and display Mean Absolute Error (MAE) and R² score
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
import numpy as np
import pandas as pd
# 5. Plot the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=['A','B','C'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A', 'B', 'C'])
# 6. Print classification report and accuracy
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
# 5. Apply the model on test and train sets. Plot prediction outcomes
import matplotlib.pyplot as plt
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(train_preds, bins=3, edgecolor='black')
plt.title("Train Predictions")
plt.subplot(1, 2, 2)
plt.hist(test_preds, bins=3, edgecolor='black')
df = pd.read_csv('income_data.csv')
print(df.head(20))
#4. Use Naive Bayes to predict high_income (Yes/No) based on education, age, occupation. 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#5. Show a bar graph comparing actual vs predicted values. 
import matplotlib.pyplot as plt
labels = ['Actual', 'Predicted']
plt.bar(labels, [sum(y_test == 'Yes'), sum(y_pred == 'Yes')])
