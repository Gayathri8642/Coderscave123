# Coderscave123
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
cancer_df = pd.read_csv('breast_cancer_data.csv')
X = cancer_df.drop('diagnosis', axis=1)  # Features
y = cancer_df['diagnosis']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:\n', confusion)
print('\nClassification Report:\n', classification_rep)
