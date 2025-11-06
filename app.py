import pandas as pd
import numpy as np
import joblib

# load the trained model and scaler
model = joblib.load('model/ufc_model.pkl')
scaler = joblib.load('model/scaler.pkl')

df = pd.read_csv('data/ufc.csv')

# normalize fighter name columns
name_cols = ['r_name', 'b_name', 'winner']
for col in name_cols:
    df[col] = df[col].str.lower().str.strip()

def get_fighter_stats(fighter_name, corner):
    # return fighter stats as a dictionary
    # fighter_name: string of fighter's full name
    # corner: 'r' for red, 'b' for blue
    fighter_row = df[df[f'{corner}_name'] == fighter_name].iloc[0]

    stats = {
        # striking accuracy
        'sig_str': fighter_row[f'{corner}_sig_str_acc'],
        # striking defense
        'str_def': fighter_row[f'{corner}_str_def'],
        # takedown accuracy
        'td': fighter_row[f'{corner}_td_acc'],
        # takedown defense
        'td_def': fighter_row[f'{corner}_td_def'],
        # submission attempts
        'sub': fighter_row[f'{corner}_sub_att'],
        # control time
        'ctrl': fighter_row[f'{corner}_ctrl'],
        # reach
        'reach': fighter_row[f'{corner}_reach'],
        # height
        'height': fighter_row[f'{corner}_height'],
        # wins
        'wins': fighter_row[f'{corner}_wins'],
        # losses
        'losses': fighter_row[f'{corner}_losses']
    }

    # replace NaN with 0
    for key in stats:
        if pd.isna(stats[key]):
            stats[key] = 0

    return stats

def predict_fight(red_name, blue_name):
    try:
        red_stats = get_fighter_stats(red_name, 'r')
        blue_stats = get_fighter_stats(blue_name, 'b')
    except IndexError:
        return "One or both fighter names not found in dataset."
    if red_name == blue_name:
        return "Please enter two different fighter names."

    # calculate differences
    features = np.array([
        red_stats['sig_str'] - blue_stats['sig_str'],
        red_stats['str_def'] - blue_stats['str_def'],
        red_stats['td'] - blue_stats['td'],
        red_stats['td_def'] - blue_stats['td_def'],
        red_stats['sub'] - blue_stats['sub'],
        red_stats['ctrl'] - blue_stats['ctrl'],
        red_stats['reach'] - blue_stats['reach'],
        red_stats['height'] - blue_stats['height'],
        (red_stats['wins'] / (red_stats['wins'] + red_stats['losses'] + 1e-5)) -
        (blue_stats['wins'] / (blue_stats['wins'] + blue_stats['losses'] + 1e-5))
    ]).reshape(1, -1)

    # scale features
    feature_names = ['sig_str_diff', 'str_def_diff', 'td_diff', 'td_def_diff', 'sub_diff', 'ctrl_diff', 'reach_diff', 'height_diff', 'wins_perc_diff']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)

    # make prediction
    prediction = model.predict(features_scaled)[0]
    # get probability
    prob_red = model.predict_proba(features_scaled)[0][1]
    prob_blue = model.predict_proba(features_scaled)[0][0]

    # map prediction to fighter name
    if prediction == 1:
        confidence = round(prob_red * 100, 2)
        return f"{red_name.title()} will likely win ({confidence}% confidence)."
    else:
        confidence = round(prob_blue * 100, 2)
        return f"{blue_name.title()} will likely win ({confidence}% confidence)."

while True:
    red = input("Enter red corner (higher ranked) fighter's name: ").lower().strip()
    blue = input("Enter blue corner (lower ranked) fighter's name: ").lower().strip()
    print(predict_fight(red, blue))

    cont = input("Predict another fight? (y/n): ")
    if cont.lower() != 'y':
        break
