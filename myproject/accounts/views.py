from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import EditProfileForm, SignUpForm
from pathlib import Path
import pandas as pd
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

# CSV Path
CSV_PATH = Path(__file__).resolve().parent.parent / "FMEL_Dataset.csv"


# ----------- Utility: Convert Matplotlib figure to base64 string ------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ----------- GLOBAL MODEL TRAINING (only once) ------------------------------
df = pd.read_csv(CSV_PATH)
df["outcome"] = df.apply(
    lambda x: 1 if x["localGoals"] > x["visitorGoals"] else (0 if x["localGoals"] == x["visitorGoals"] else -1),
    axis=1
)

le_season = LabelEncoder()
le_local = LabelEncoder()
le_visitor = LabelEncoder()
df["season_enc"] = le_season.fit_transform(df["season"])
df["localTeam_enc"] = le_local.fit_transform(df["localTeam"])
df["visitorTeam_enc"] = le_visitor.fit_transform(df["visitorTeam"])

X = df[["season_enc", "division", "round", "localTeam_enc", "visitorTeam_enc"]]
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Performance metrics
metrics = {
    "accuracy_train": accuracy_score(y_train, y_train_pred),
    "accuracy_test": accuracy_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
}

# Feature importance chart
fig1 = plt.figure()
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
feature_chart = fig_to_base64(fig1)

# Train vs Test chart
fig2 = plt.figure()
plt.bar(["Train", "Test"], [metrics["accuracy_train"], metrics["accuracy_test"]])
plt.ylim(0, 1)
plt.title("Train vs Test Accuracy")
train_test_chart = fig_to_base64(fig2)

# ROC Curve (OvR)
proba = model.predict_proba(X_test)
fig3 = plt.figure()
for i, label in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve((y_test == label).astype(int), proba[:, i])
    plt.plot(fpr, tpr, label=f"Class {label} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.title("ROC Curve (OvR)")
roc_chart = fig_to_base64(fig3)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
fig4 = plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
confusion_matrix_chart = fig_to_base64(fig4)


# ------------------- Django Views ------------------------------------------
def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request,'Account created successfully!')
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request,'accounts/signup.html',{'form':form})

def login_view(request):
    if request.method=='POST':
        form = AuthenticationForm(request,data=request.POST)
        if form.is_valid():
            username=form.cleaned_data.get('username')
            password=form.cleaned_data.get('password')
            user=authenticate(username=username,password=password)
            if user:
                login(request,user)
                return redirect('dashboard')
        messages.error(request,'Invalid username or password')
    else:
        form = AuthenticationForm()
    return render(request,'accounts/login.html',{'form':form})

def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    prediction = None
    if request.method == "POST":
        season = request.POST.get("season")
        division = request.POST.get("division")
        round_ = request.POST.get("round")
        localTeam = request.POST.get("localTeam")
        visitorTeam = request.POST.get("visitorTeam")

        # Convert safely
        try:
            division = int(division) if division else 0
            round_ = int(round_) if round_ else 0
        except ValueError:
            division, round_ = 0, 0

        x = {
            "season_enc": le_season.transform([season])[0],
            "division": division,
            "round": round_,
            "localTeam_enc": le_local.transform([localTeam])[0],
            "visitorTeam_enc": le_visitor.transform([visitorTeam])[0],
        }
        pred = model.predict(pd.DataFrame([x]))[0]
        mapping = {1: "🏠 Home Win", 0: "🤝 Draw", -1: "🛫 Away Win"}
        prediction = mapping[pred]

    context = {
        "metrics": metrics,
        "feature_chart": feature_chart,
        "train_test_chart": train_test_chart,
        "roc_chart": roc_chart,
        "confusion_matrix_chart": confusion_matrix_chart,
        "teams": df["localTeam"].unique(),
        "seasons": df["season"].unique(),
        "divisions": sorted(df["division"].unique()),
        "prediction": prediction,
    }
    return render(request, "dashboard.html", context)


@login_required
def performance_view(request):
    context = {
        "metrics": metrics,
        "feature_chart": feature_chart,
        "roc_chart": roc_chart,
        "train_test_chart": train_test_chart,
        "confusion_matrix_chart": confusion_matrix_chart,
    }
    return render(request, "performance.html", context)
@login_required
def edit_profile(request):
    user = request.user
    if request.method == "POST":
        form = EditProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('dashboard')
    else:
        form = EditProfileForm(instance=user)
    return render(request, "accounts/edit_profile.html", {"form": form})
@login_required
def prediction_view(request):
    prediction = None
    if request.method == "POST":
        # Get input from form
        home_elo = float(request.POST['home_elo'])
        away_elo = float(request.POST['away_elo'])

        # Simple Random Forest from scratch example (actually just rules)
        votes = []
        votes.append('Home' if home_elo > away_elo else 'Away')
        votes.append('Home' if home_elo - away_elo > 50 else 'Draw')
        votes.append('Home' if home_elo > away_elo else 'Away')

        # Majority vote
        prediction = max(set(votes), key=votes.count)

    return render(request, "prediction.html", {"prediction": prediction})
def fixtures_view(request):
    fixtures = [
        {"date": "2025-09-10", "time": "18:00", "home": "Arsenal", "away": "Chelsea", "venue": "Emirates Stadium"},
        {"date": "2025-09-11", "time": "20:00", "home": "Barcelona", "away": "Real Madrid", "venue": "Camp Nou"},
    ]
    return render(request, 'fixtures.html', {"fixtures": fixtures})


def table_view(request):
    league_table = [
        {"pos": 1, "team": "Manchester City", "played": 5, "won": 4, "draw": 1, "lost": 0, "gf": 12, "ga": 4, "gd": 8, "points": 13},
        {"pos": 2, "team": "Liverpool", "played": 5, "won": 4, "draw": 0, "lost": 1, "gf": 10, "ga": 5, "gd": 5, "points": 12},
    ]
    return render(request, 'table.html', {"league_table": league_table})
from django.shortcuts import render
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

CSV_PATH = Path(__file__).resolve().parent.parent / "FMEL_Dataset.csv"

import json
from sklearn.metrics import roc_curve, auc

def analysis_view(request):
    # Feature stats from the real split
    features = list(X.columns)
    train_means = X_train.mean().round(2).tolist()
    test_means = X_test.mean().round(2).tolist()

    # ROC (OvR)
    proba = model.predict_proba(X_test)
    roc_data = {}
    for i, label in enumerate(model.classes_):
        fpr, tpr, _ = roc_curve((y_test == label).astype(int), proba[:, i])
        if len(fpr) > 0 and len(tpr) > 0:
            roc_data[str(label)] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(auc(fpr, tpr), 2),
            }

    context = {
        "features_json": json.dumps(features),
        "train_means_json": json.dumps(train_means),
        "test_means_json": json.dumps(test_means),
        "roc_data_json": json.dumps(roc_data),
    }
    return render(request, "analysis.html", context)
