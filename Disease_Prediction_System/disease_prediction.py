import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (You can merge both CSV files if needed, as described above)
df = pd.read_csv('merged_dataset.csv')

# Display the first few rows to understand its structure
print(df.head())

# Preprocessing: Remove any irrelevant columns or handle missing data
df.dropna(inplace=True)  # Remove rows with missing values



# Split data into features (X) and target (y)
X = df.drop(columns=['Disease'])  
y = df['Disease']  

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using Random Forest Classifier Model for simplicity
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display confusion matrix for more insights
print(confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# Print classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

# Save the model for future predictions
import joblib
joblib.dump(model, 'disease_prediction_model.pkl')
