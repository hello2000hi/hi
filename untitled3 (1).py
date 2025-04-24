# 1. Create a NumPy array of shape (4, 3) filled with zeros
import numpy as np
arr = np.zeros((4, 3))
print(arr)

# 2. Change the values of the first row to ones
arr[0] = 1
print(arr)

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
print(df)

# 1. Load hospital_data.csv and print first 12 rows
df = pd.read_csv('hospital_data.csv')
print(df.head(12))

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


# 1. Create a NumPy array of 10 linearly spaced values between 0 and 1
import numpy as np
arr = np.linspace(0, 1, 10)
print(arr)

# 2. Reshape it into a 2x5 array
reshaped = arr.reshape(2, 5)
print(reshaped)

# 3. Create a DataFrame using random numbers for 3 columns and 5 rows
import pandas as pd
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
print(df)

# 4. Filter the rows where column A > column B
filtered = df[df['A'] > df['B']]
print(filtered)

# 1. Load loan_data.csv and print first 8 records
df = pd.read_csv('loan_data.csv')
print(df.head(8))

# 2. Add a column emi using formula: loan_amount / loan_term
df['emi'] = df['loan_amount'] / df['loan_term']

# 3. Split the data into training and testing parts
from sklearn.model_selection import train_test_split
X = df[['income', 'credit_score', 'loan_amount']]
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Build a KNN model to predict loan_status
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Use matplotlib to visualize prediction outcomes
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
plt.scatter(range(len(y_test)), y_test, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', marker='x')
plt.legend()
plt.title("Prediction Outcomes")
plt.show()

# 6. Evaluate the model using confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Section A

import numpy as np
import pandas as pd

# 1. Create a NumPy array using the range function
array = np.array(range(10))
print("NumPy Array:", array)

# 2. Print the data type of each element within the array
print("Data types of elements:")
for element in array:
    print(type(element))

# 3. Create a pandas dataframe from the python dictionary for Car Record
car_data = {
    'Brand': ['Toyota', 'Honda', 'Ford'],
    'Model': ['Corolla', 'Civic', 'F-150'],
    'Year': [2010, 2012, 2015]
}
car_df = pd.DataFrame(car_data)
print("\nCar DataFrame:")
print(car_df)

# 4. Add one column in the above dataframe
car_df['Price'] = [8000, 9000, 15000]
print("\nUpdated Car DataFrame:")
print(car_df)

# Section B

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load the fracture.csv data into pandas dataframe and print the first 15 records
df = pd.read_csv("fracture.csv")
print("First 15 Records:")
print(df.head(15))

# 2. Add a new column named “bmi” using the formula
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 🔁 Convert 'sex' and 'fracture' columns to numeric
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 3. Split the data set into test and train
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Apply the model on to the test and train data and plot prediction outcomes
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plotting predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, color='blue')
plt.title('Train Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='green')
plt.title('Test Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# 6. Calculate accuracy using confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)
print("\nConfusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(f"Accuracy: {acc:.2f}")

#set2
# Section A

import numpy as np
import pandas as pd

# 1. Create a 3x3 NumPy array with values between 0 to 9
array = np.arange(9).reshape(3, 3)
print("3x3 NumPy Array:")
print(array)

# 2. Print the data type of element of the array
print("\nData type of array elements:")
print(array.dtype)

# 3. Create a pandas dataframe for Mobile Phone Details
mobile_data = {
    'Brand': ['Samsung', 'Apple', 'OnePlus'],
    'Model': ['Galaxy S23', 'iPhone 14', 'OnePlus 11'],
    'Price': [70000, 80000, 60000]
}
df_mobile = pd.DataFrame(mobile_data)
print("\nMobile Phone DataFrame:")
print(df_mobile)

# 4. Add one more column to the above dataframe
df_mobile['Storage'] = ['128GB', '256GB', '256GB']
print("\nDataFrame after adding 'Storage' column:")
print(df_mobile)

# Section B

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load the fracture.csv and display the last five records
df = pd.read_csv("fracture.csv")
print("Last 5 Records:")
print(df.tail())

# 2. Add BMI column: BMI = weight_kg / (height_cm/100)^2
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# Convert 'sex' and 'fracture' to numeric
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 3. Split dataset into train/test (70:30)
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Build SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 5. Predict and plot outcomes
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, color='orange')
plt.title('Train Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='purple')
plt.title('Test Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# 6. Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)
print("\nConfusion Matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(f"\nAccuracy: {acc:.2f}")

# SET-03 – Section A

import numpy as np
import pandas as pd

# 1. Create a NumPy array with elements that are multiples of 5
array = np.arange(5, 55, 5)
print("NumPy array with multiples of 5:")
print(array)

# 2. Print the standard deviation of the array
std_dev = np.std(array)
print("\nStandard Deviation:", std_dev)

# 3. Create a pandas dataframe for Book details
book_data = {
    'Title': ['Python 101', 'Data Science Handbook', 'AI Basics'],
    'Author': ['John Doe', 'Jane Smith', 'Alice Brown'],
    'Price': [250, 500, 300]
}
df_books = pd.DataFrame(book_data)
print("\nBook Details DataFrame:")
print(df_books)

# 4. Rename one column
df_books.rename(columns={'Price': 'Cost'}, inplace=True)
print("\nDataFrame after renaming 'Price' to 'Cost':")
print(df_books)

#✅ SET-03 – Section B

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load the fracture.csv into DataFrame
df = pd.read_csv("fracture.csv")
print("Fracture Data (First 5 rows):")
print(df.head())

# 2. Add a new column 'bmi'
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# Encode categorical variables
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 3. Split the data into 80% train, 20% test
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 5. Predict and plot
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='teal', alpha=0.6)
plt.title('Train Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='orange', alpha=0.6)
plt.title('Test Data Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# 6. Confusion matrix & accuracy
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(f"\nModel Accuracy: {acc:.2f}")

# SET-04 – Section A

import numpy as np
import pandas as pd

# 1. Create a NumPy array from a list
my_list = [12, 45, 67, 89, 23, 56]
np_array = np.array(my_list)
print("NumPy Array:")
print(np_array)

# 2. Print max and min values
print("\nMaximum value:", np.max(np_array))
print("Minimum value:", np.min(np_array))

# 3. Create pandas dataframe from Student Record dictionary
student_data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [20, 21, 19],
    'Grade': ['A', 'B', 'A']
}
df_students = pd.DataFrame(student_data)
print("\nStudent Record DataFrame:")
print(df_students)

# 4. Delete one column from the DataFrame
df_students.drop(columns='Grade', inplace=True)
print("\nDataFrame after deleting 'Grade' column:")
print(df_students)

# SET-04 – Section B

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load fracture.csv and print first 15 records
df = pd.read_csv("fracture.csv")
print("First 15 Records:")
print(df.head(15))

# 2. Add 'bmi' column
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 3. Encode 'sex' and 'fracture' for ML model
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# Split dataset
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict and plot outcomes
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
plt.title("Train Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5)
plt.title("Test Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

# 6. Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(f"\nAccuracy of Logistic Regression Model: {acc:.2f}")

#SET-05 – Section A

import numpy as np
import pandas as pd

# 1. Create a 5x5 NumPy array with values from 0 to 24
array_5x5 = np.arange(25).reshape(5, 5)
print("5x5 NumPy Array:")
print(array_5x5)

# 2. Print the size and shape of the array
print("\nArray Size:", array_5x5.size)
print("Array Shape:", array_5x5.shape)

# 3. Create a pandas dataframe for Mobile phone details
mobile_data = {
    'Brand': ['Samsung', 'Apple', 'Xiaomi'],
    'Model': ['S21', 'iPhone 13', 'Redmi Note 10'],
    'Price': [70000, 80000, 15000]
}
df_mobiles = pd.DataFrame(mobile_data)
print("\nMobile Phone DataFrame:")
print(df_mobiles)

# 4. Add one more column (e.g., 'RAM')
df_mobiles['RAM'] = ['8GB', '6GB', '4GB']
print("\nDataFrame after adding RAM column:")
print(df_mobiles)

#SET-05 – Section B

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load fracture.csv
df = pd.read_csv("fracture.csv")
print("Last 5 Records:")
print(df.tail())

# 2. Add BMI column
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 3. Encode categorical data
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 4. Prepare data
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Build KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict
y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)

# Plot predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6, color='blue')
plt.title("Train Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6, color='orange')
plt.title("Test Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

# 6. Confusion matrix & accuracy
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(f"\nKNN Model Accuracy: {acc:.2f}")

#SET-06 – Section A

import numpy as np
import pandas as pd

# 1. Create a NumPy array using random function
random_array = np.random.rand(5, 5)
print("Random NumPy Array (5x5):")
print(random_array)

# 2. Print the average value in the above array
average_value = np.mean(random_array)
print("\nAverage value of the array:", average_value)

# 3. Create a pandas dataframe from a Python dictionary for Book details
book_data = {
    'Title': ['The Alchemist', '1984', 'To Kill a Mockingbird'],
    'Author': ['Paulo Coelho', 'George Orwell', 'Harper Lee'],
    'Price': [350, 400, 300]
}
df_books = pd.DataFrame(book_data)
print("\nBook DataFrame:")
print(df_books)

# 4. Rename one column from the above dataframe (e.g., rename 'Price' to 'Cost')
df_books.rename(columns={'Price': 'Cost'}, inplace=True)
print("\nRenamed DataFrame:")
print(df_books)

#SET-06 – Section B

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Load fracture.csv
df = pd.read_csv("fracture.csv")
print("First 5 Records:")
print(df.head())

# 2. Add new column 'bmi'
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 3. Encode 'sex' and 'fracture'
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 4. Split the data in 20:80 ratio
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predictions
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

# Plotting predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
plt.title("Train Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5)
plt.title("Test Data Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

# 6. Confusion Matrix & Accuracy
cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

print(f"\nSVM Model Accuracy: {acc:.2f}")

# 1. Create a 2D NumPy array using np.arange() and reshape()
import numpy as np

array_2d = np.arange(12).reshape(3, 4)
print(array_2d)

# 2. Access the last row and first column from the array
print("Last row:", array_2d[-1])
print("First column:", array_2d[:, 0])

# 3. Create a DataFrame from a nested dictionary
import pandas as pd

nested_dict = {
    'Math': {'Tom': 85, 'Jerry': 90},
    'Science': {'Tom': 78, 'Jerry': 88}
}
df = pd.DataFrame(nested_dict)
print(df)

# 4. Rename one of the columns
df.rename(columns={'Math': 'Mathematics'}, inplace=True)
print(df)

# 1. Load the sales.csv file and display the first 10 rows
sales_df = pd.read_csv('sales.csv')
print(sales_df.head(10))

# 2. Add a new column discounted_price
sales_df['discounted_price'] = sales_df['price'] - (sales_df['price'] * sales_df['discount'] / 100)
print(sales_df.head())

# 3. Split the dataset into training and testing datasets
from sklearn.model_selection import train_test_split

X = sales_df[['advertising', 'discount', 'price']]
y = sales_df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a Linear Regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. Plot actual vs predicted values
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# 6. Calculate and display Mean Absolute Error (MAE) and R² score
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("MAE:", mae)
print("R² Score:", r2)

import numpy as np
import pandas as pd

# 1. Create a NumPy array of random integers between 10 and 50
arr = np.random.randint(10, 51, size=10)
print("Array:", arr)

# 2. Calculate mean and standard deviation
print("Mean:", np.mean(arr))
print("Std Dev:", np.std(arr))

# 3. Create a pandas DataFrame using pd.read_csv() on sample data
df = pd.read_csv('sample_data.csv')
print(df.head())

# 4. Sort the DataFrame based on one of its numeric columns
sorted_df = df.sort_values(by=df.select_dtypes(include='number').columns[0])
print(sorted_df.head())

# 1. Load student_scores.csv and print last 10 records
df = pd.read_csv('student_scores.csv')
print(df.tail(10))

# 2. Add grade column based on score
def get_grade(score):
    if score > 90:
        return 'A'
    elif score > 75:
        return 'B'
    else:
        return 'C'
df['grade'] = df['score'].apply(get_grade)
print(df.head())

# 3. Use train_test_split to prepare data
from sklearn.model_selection import train_test_split
X = df[['age', 'study_hours', 'attendance']]
y = df['grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Plot the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=['A','B','C'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A', 'B', 'C'])
disp.plot()
plt.show()

# 6. Print classification report and accuracy
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1. Create a NumPy array from a list
import numpy as np

arr = np.array([5, 2, 9, 1, 7])
print("Array:", arr)

# 2. Print the maximum and minimum value in the above NumPy array
print("Max:", np.max(arr))
print("Min:", np.min(arr))

# 3. Create a pandas DataFrame from a Python dictionary
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print("\nDataFrame:\n", df)

# 4. Delete one column from the above DataFrame
df = df.drop(columns=['Age'])
print("\nDataFrame after deleting 'Age' column:\n", df)

# 1. Load the fracture.csv data into a pandas DataFrame and print the first 15 records
fracture_df = pd.read_csv('fracture.csv')  # Ensure this file exists in the same folder
print(fracture_df.head(15))

# 2. Add a new column named “bmi” to store Body Mass Index
fracture_df['bmi'] = fracture_df['weight'] / (fracture_df['height'] ** 2)
print(fracture_df[['weight', 'height', 'bmi']].head())

# 3. Split the dataset into test and train sets
from sklearn.model_selection import train_test_split

X = fracture_df[['age', 'sex', 'bmi', 'bmd']]
y = fracture_df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a logistic regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

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
plt.title("Test Predictions")
plt.tight_layout()
plt.show()

# 6. Calculate accuracy using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, test_preds)
accuracy = accuracy_score(y_test, test_preds)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)

# 1. Create a NumPy array of even numbers using list comprehension
import numpy as np

even_arr = np.array([x for x in range(2, 21, 2)])
print(even_arr)

# 2. Slice to get first 5 elements
print(even_arr[:5])

# 3. Create a pandas DataFrame from list of tuples
import pandas as pd

data = [('John', 25), ('Anna', 28)]
df = pd.DataFrame(data, columns=['Name', 'Age'])
print(df)

# 4. Add new column
df['Country'] = ['USA', 'Canada']
print(df)

df = pd.read_csv('income_data.csv')
print(df.head(20))

#2. Create a new column tax = income * 0.15 

df['tax'] = df['income'] * 0.15
#3. Split the data into train and test datasets. 
from sklearn.model_selection import train_test_split
X = df[['education', 'age', 'occupation']]
y = df['high_income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#4. Use Naive Bayes to predict high_income (Yes/No) based on education, age, occupation. 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#5. Show a bar graph comparing actual vs predicted values. 
import matplotlib.pyplot as plt
labels = ['Actual', 'Predicted']
plt.bar(labels, [sum(y_test == 'Yes'), sum(y_pred == 'Yes')])
plt.title("Actual vs Predicted High Income")
plt.show()

#6. Compute confusion matrix and classification report. 
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))