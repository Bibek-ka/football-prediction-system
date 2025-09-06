import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("FMEL_Dataset.csv")

# Outcome: 1=home win, 0=draw, -1=away win -> 0,1,2 for ML
df["outcome"] = df.apply(lambda row: 1 if row["localGoals"] > row["visitorGoals"] else (-1 if row["localGoals"] < row["visitorGoals"] else 0), axis=1)
df["outcome_class"] = df["outcome"] + 1  # Classes: 0=away win,1=draw,2=home win
df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

# 2. Compute league table stats (points_diff & gd_diff)
def compute_league_table(df):
    points, goals_for, goals_against = defaultdict(int), defaultdict(int), defaultdict(int)
    local_pts, visitor_pts, local_gd, visitor_gd = [], [], [], []

    for _, row in df.iterrows():
        home, away = row["localTeam"], row["visitorTeam"]
        hg, ag = row["localGoals"], row["visitorGoals"]

        # Save current points & GD before update
        local_pts.append(points[home])
        visitor_pts.append(points[away])
        local_gd.append(goals_for[home] - goals_against[home])
        visitor_gd.append(goals_for[away] - goals_against[away])

        # Update goals
        goals_for[home] += hg
        goals_against[home] += ag
        goals_for[away] += ag
        goals_against[away] += hg

        # Update points
        if hg > ag:
            points[home] += 3
        elif hg < ag:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    df["local_points"] = local_pts
    df["visitor_points"] = visitor_pts
    df["points_diff"] = df["local_points"] - df["visitor_points"]
    df["local_gd"] = local_gd
    df["visitor_gd"] = visitor_gd
    df["gd_diff"] = df["local_gd"] - df["visitor_gd"]
    return df

df = compute_league_table(df)

# 3. Compute Elo ratings
def compute_elo(df, k=20):
    elo_ratings = defaultdict(lambda: 1500)
    local_elo, visitor_elo = [], []
    for _, row in df.iterrows():
        home, away = row["localTeam"], row["visitorTeam"]
        R_home, R_away = elo_ratings[home], elo_ratings[away]
        E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))

        if row["localGoals"] > row["visitorGoals"]:
            S_home, S_away = 1, 0
        elif row["localGoals"] < row["visitorGoals"]:
            S_home, S_away = 0, 1
        else:
            S_home, S_away = 0.5, 0.5

        local_elo.append(R_home)
        visitor_elo.append(R_away)

        elo_ratings[home] += k * (S_home - E_home)
        elo_ratings[away] += k * (S_away - (1 - E_home))

    df["local_elo"] = local_elo
    df["visitor_elo"] = visitor_elo
    df["elo_diff"] = df["local_elo"] - df["visitor_elo"]
    return df

df = compute_elo(df)

# 4. Add home advantage feature
df["home_advantage"] = 1

# 5. Features and target
features = [
    "local_elo", "visitor_elo", "elo_diff",
    "local_points", "visitor_points", "points_diff",
    "gd_diff",
    "home_advantage"
]
X = df[features]
y = df["outcome_class"]

# 6. Check class distribution
print("Class distribution:\n", y.value_counts(normalize=True))

# 7. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train XGBoost classifier
model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Feature importance plot
plt.barh(features, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.show()
