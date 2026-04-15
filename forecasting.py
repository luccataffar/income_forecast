import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------
df = pd.read_csv("synthetic_pp_mrp_cases.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")


# -----------------------------------------------------------
# 2. Time features
# -----------------------------------------------------------
df["month"] = df["date"].dt.month

df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)


# -----------------------------------------------------------
# 3. Lag features
# -----------------------------------------------------------
for lag in [1,2,3,6,12]:
    df[f"lag{lag}"] = df["cases"].shift(lag)


# -----------------------------------------------------------
# 4. Rolling features
# -----------------------------------------------------------
df["rolling3"] = df["cases"].shift(1).rolling(3).mean()
df["rolling6"] = df["cases"].shift(1).rolling(6).mean()
df["rolling12"] = df["cases"].shift(1).rolling(12).mean()


# -----------------------------------------------------------
# 5. Trend
# -----------------------------------------------------------
df["time_index"] = np.arange(len(df))


# -----------------------------------------------------------
# 6. Remove NA
# -----------------------------------------------------------
df = df.dropna()


# -----------------------------------------------------------
# 7. Train/Test split
# -----------------------------------------------------------
train = df.iloc[:-12].copy()
test = df.iloc[-12:].copy()

features = [c for c in df.columns if c not in ["cases","date","component"]]

X_train = train[features]
y_train = train["cases"]


# -----------------------------------------------------------
# 8. Train Random Forest
# -----------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------------------------------------
# 9. Recursive forecasting
# -----------------------------------------------------------
history = train.copy()
predictions = []

for i in range(12):

    row = test.iloc[i].copy()

    # atualizar lags
    row["lag1"] = history["cases"].iloc[-1]
    row["lag2"] = history["cases"].iloc[-2]
    row["lag3"] = history["cases"].iloc[-3]
    row["lag6"] = history["cases"].iloc[-6]
    row["lag12"] = history["cases"].iloc[-12]

    # atualizar rolling
    row["rolling3"] = history["cases"].iloc[-3:].mean()
    row["rolling6"] = history["cases"].iloc[-6:].mean()
    row["rolling12"] = history["cases"].iloc[-12:].mean()

    # manter feature names
    X_pred = pd.DataFrame([row[features]])

    pred = model.predict(X_pred)[0]

    predictions.append(pred)

    new_row = row.copy()
    new_row["cases"] = pred

    history = pd.concat([history, pd.DataFrame([new_row])])


predictions = np.array(predictions)


# -----------------------------------------------------------
# 10. MAE
# -----------------------------------------------------------
mae = mean_absolute_error(test["cases"], predictions)

print("MAE (last 12 months):", round(mae,2))


# -----------------------------------------------------------
# 11. Chart
# -----------------------------------------------------------
plt.figure(figsize=(14,6))

# Plot actual values
plt.plot(
    test["date"],
    test["cases"],
    label="Actual",
    linewidth=2.5,
    marker="o"
)

# Plot predictions
plt.plot(
    test["date"],
    predictions,
    label="Forecast",
    linestyle="--",
    linewidth=2.5,
    marker="o"
)

# --- VALUE LABELS ---
for i in range(len(test)):
    plt.text(
        test["date"].iloc[i],
        predictions[i] + 2,
        f"{int(predictions[i])}",
        ha="center",
        fontsize=8,
        alpha=0.8
    )

# --- METRICS ---
mape = np.mean(np.abs((test["cases"] - predictions) / test["cases"])) * 100

plt.title("Forecast Accuracy — Last 12 Months", fontsize=14)

plt.text(
    test["date"].iloc[len(test)//2],
    max(test["cases"]) * 0.97,
    f"MAE: {round(mae,1)}  |  Error: {round(mape,1)}%",
    fontsize=11
)

# Axis labels
plt.xlabel("Date")
plt.ylabel("Cases")

# legend
plt.legend()

# Rotate dates
plt.xticks(rotation=45)

# Remove visual exaggeration
plt.ylim(
    min(test["cases"].min(), predictions.min()) - 10,
    max(test["cases"].max(), predictions.max()) + 10
)

plt.tight_layout()
plt.show()
