import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model

try:
    model = CatBoostClassifier()
    model.load_model("genshin_artifact_rater.cbm")
    print("Loaded successfully!\n")
except Exception as e:
    print("Error: Could not load 'genshin_artifact_rater.cbm'.")
    print("Make sure you ran the training script and saved the model first!")
    exit()

# Define the Mapping (Number -> Text)

score_map_reverse = {
    0: 'Complete trash',
    1: 'Trash',
    2: 'Usable',
    3: 'Good',
    4: 'Excellent',
    5: 'Marvelous'
}


# Collect User Input

artifact_data = {
    'Set': 'Golden Troupe',
    'Type': 'Circlet',
    'Main_Stat': 'Healing',
    'Number_of_substat': 3,
    'Percent_ATK': 0,  
    'Percent_HP': 1,
    'Percent_DEF': 0,
    'ATK': 0,
    'HP': 0,
    'DEF': 0,
    'ER': 0,           
    'EM': 0,
    'Crit_Rate': 1,    
    'Crit_DMG': 1      
}

# Create DataFrame & Predict

df_new = pd.DataFrame([artifact_data])

# Ensure columns are in the exact same order as training
feature_order = [
    'Set', 'Type', 'Main_Stat', 'Number_of_substat', 
    'Percent_ATK', 'Percent_HP', 'Percent_DEF', 
    'ATK', 'HP', 'DEF', 'ER', 'EM', 
    'Crit_Rate', 'Crit_DMG'
]
df_new = df_new[feature_order]

# Get the prediction (returns a list of lists, we want the first one)
prediction_id = model.predict(df_new)[0]
prediction_id = int(prediction_id[0]) 

predicted_label = score_map_reverse.get(prediction_id, "Unknown")


print("\n" + "="*30)
print(f"PREDICTED SCORE: {predicted_label.upper()}")
print("="*30)

# Show Probability (Confidence)
probs = model.predict_proba(df_new)[0]
confidence = probs[prediction_id] * 100
print(f"(Confidence: {confidence:.1f}%)")