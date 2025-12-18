import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier


db_connection_str = 'postgresql://postgres@localhost/postgres'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM "Artifact_itself"', db_connection)


df = df[df['Score'] != 'Unknown']

score_map = {
    'Complete trash': 0,
    'Trash': 1,
    'Usable': 2,
    'Good': 3,
    'Excellent': 4,
    'Marvelous': 5
}
df['Target_Score'] = df['Score'].map(score_map)
df = df.dropna(subset=['Target_Score'])

feature_cols = [
    'Set', 'Type', 'Main_Stat', 'Number_of_substat', 
    'Percent_ATK', 'Percent_HP', 'Percent_DEF', 
    'ATK', 'HP', 'DEF', 'ER', 'EM', 
    'Crit_Rate', 'Crit_DMG'
]

X = df[feature_cols]
y = df['Target_Score']
cat_features = ['Set', 'Type', 'Main_Stat']


model = CatBoostClassifier(
    iterations=2000,   
    learning_rate=0.02,
    depth=12,
    loss_function='MultiClass',
    
    # Add class weights to focus on rarer classes
    auto_class_weights='Balanced', 
    
    verbose=200,
    task_type="GPU"
)


print("Training on 100% of data (Full Production Mode)...")
model.fit(X, y, cat_features=cat_features)


model.save_model("genshin_artifact_rater_full.cbm")
print("Model saved as 'genshin_artifact_rater_full.cbm'")
print("You can now use this file in your 'rate_my_artifact.py' script.")