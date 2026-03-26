import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("../data/sample_sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['dayofweek'] = df['Date'].dt.dayofweek

# Encode product
df = pd.get_dummies(df, columns=['Product'])

# Split
train = df[df['Date'] < '2025-01-04']
test = df[df['Date'] >= '2025-01-04']

X_train = train.drop(['Sales', 'Date'], axis=1)
y_train = train['Sales']

X_test = test.drop(['Sales', 'Date'], axis=1)
y_test = test['Sales']

# Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)

# Save output
output = test.copy()
output['Predicted'] = preds
output.to_csv("../outputs/forecast_results.csv", index=False)
