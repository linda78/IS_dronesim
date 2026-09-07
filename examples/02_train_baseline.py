"""Train a baseline model on the tabular features.

Run example 01 first, it writes the dataset this script loads:

    python examples/01_generate_and_plot.py
    python examples/02_train_baseline.py

This is only meant to show the path from `df` to a model. The MLP and the CNN of your
assignment are implemented in PyTorch, this baseline deliberately is not.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from dronesim import FEATURE_NAMES, load_dataset

df, scans = load_dataset("data/train")
print(f"loaded {len(df)} samples, collision rate {df['collided'].mean():.1%}")

X = df[FEATURE_NAMES]
y = df["collided"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\naccuracy: {model.score(X_test, y_test):.3f}")
print(f"f1:       {f1_score(y_test, y_pred):.3f}\n")
print(classification_report(y_test, y_pred, target_names=["free", "collision"]))

print("feature importances:")
for name, importance in sorted(
    zip(FEATURE_NAMES, model.feature_importances_), key=lambda p: -p[1]
):
    print(f"  {name:14s} {importance:.3f}")
