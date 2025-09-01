import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Try to read CSV with some robustness against bad quotes and malformed lines
try:
    df = pd.read_csv('combined_output.csv', sep=';', encoding='ISO-8859-1', engine='python', quoting=3, error_bad_lines=False)
    print("CSV loaded!")
    print(df.head())
except Exception as e:
    print(f"Failed to load CSV: {e}")

print("CSV loaded successfully!")
print(df.head())
print(df.info())
print(df.isnull().sum())

# Drop missing values or handle them
df = df.dropna()

# Drop irrelevant columns if any (example)
cols_to_drop = ['Unnamed: 10']  # Adjust as needed
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# Encode categorical variables
for col in ['team', 'position', 'nation']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Create target variable from win/draw/loss indicators
def get_match_outcome(row):
    if row.get('win', 0) == 1:
        return 'win'
    elif row.get('draw', 0) == 1:
        return 'draw'
    elif row.get('loss', 0) == 1:
        return 'loss'
    else:
        return 'unknown'

df['match_outcome'] = df.apply(get_match_outcome, axis=1)

# Remove unknown outcomes
df = df[df['match_outcome'] != 'unknown']

# Prepare features and labels
X = df.drop(columns=['win', 'loss', 'draw', 'match_outcome'])
y = df['match_outcome']

# Encode any remaining object-type columns in X
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print("Feature data types:")
print(X.dtypes)

# Split dataset with stratify for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
