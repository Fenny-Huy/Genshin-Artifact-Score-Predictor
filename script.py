import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


db_connection_str = 'postgresql://postgres@localhost/postgres'
db_connection = create_engine(db_connection_str)

# Load raw data
df = pd.read_sql('SELECT * FROM "Artifact_itself"', db_connection)



# Remove 'Unknown'
df = df[df['Score'] != 'Unknown']

# Map scores to numbers strictly for calculation purposes

score_map = {
    'Complete trash': 0,
    'Trash': 1,
    'Usable': 2,
    'Good': 3,
    'Excellent': 4,
    'Marvelous': 5
}
df['Target_Score'] = df['Score'].map(score_map)
df = df.dropna(subset=['Target_Score']) # Safety check



feature_cols = [
    # Categorical Context
    'Set', 
    'Type', 
    'Main_Stat', 
    
    # Numerical Context
    'Number_of_substat', 
    
    # Binary Flags (The 0 or 1 fields)
    'Percent_ATK', 'Percent_HP', 'Percent_DEF', 
    'ATK', 'HP', 'DEF', 'ER', 'EM', 
    'Crit_Rate', 'Crit_DMG'
]

X = df[feature_cols]
y = df['Target_Score']

# Identify text columns
cat_features = ['Set', 'Type', 'Main_Stat']

# Train the Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost for finding non-linear patterns.

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    task_type="GPU"
)

print("Training on your historical decisions...")
model.fit(X_train, y_train, cat_features=cat_features)

# Review Results

print("\nEvaluation:")
preds = model.predict(X_test)
print(classification_report(
    y_test, 
    preds, 
    labels=list(score_map.values()),
    target_names=list(score_map.keys())
))

# Show what factors the data says are important.
print("\nTop Features driving YOUR scores:")
importances = model.get_feature_importance(prettified=True)
print(importances.head(10))

model.save_model("genshin_artifact_rater.cbm")