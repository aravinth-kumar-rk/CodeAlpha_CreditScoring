import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate dataset with realistic financial rules
np.random.seed(42)
n = 1000
income = np.random.randint(30000, 150000, n)
debts = np.random.randint(5000, 50000, n)
payment_history = np.random.randint(0, 2, n) # 1 = Good, 0 = Bad

# Rule: Creditworthy (1) if payment history is good AND debt is less than 50% of income
target = ((payment_history == 1) & (debts < income * 0.5)).astype(int)

df = pd.DataFrame({
    'income': income,
    'debts': debts,
    'payment_history': payment_history,
    'target': target
})

# 2. Split data (keeping DataFrame format to maintain feature names cleanly)
X = df[['income', 'debts', 'payment_history']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 4. Evaluate Performance
y_pred = dt_model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"Decision Tree Accuracy: {acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
