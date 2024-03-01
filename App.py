import streamlit as st
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder


data = pd.read_csv('Cleaned_Drug.csv')

data = data.drop(columns=['Unnamed: 0'])

X = data.drop('Satisfaction', axis=1)
y = data['Satisfaction']

label_encoder_condition = LabelEncoder()
label_encoder_drug = LabelEncoder()

original_values_condition = X['Condition'].copy()
original_values_drug = X['Drug'].copy()

X['Condition'] = label_encoder_condition.fit_transform(X['Condition'])
X['Drug'] = label_encoder_drug.fit_transform(X['Drug'])

mapping_condition = dict(zip(X['Condition'].unique(), original_values_condition.unique()))
mapping_drug = dict(zip(X['Drug'].unique(), original_values_drug.unique()))

original_encoded_features = pd.get_dummies(data[['Type', 'Indication']], drop_first=True, dtype=int)

original_encoded_features = original_encoded_features.rename(columns={'Indication_On Label':'Indication_On_Label'})

X = X.drop(columns=['Type','Indication'])

X = pd.concat([X, original_encoded_features], axis=1)

original_type_columns = original_encoded_features.filter(like='Type').columns
original_indication_columns = original_encoded_features.filter(like='Indication').columns

# Standardize the entire dataset
scaler = StandardScaler()
numerical_cols = ['Effective', 'EaseOfUse']

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, _ in kf.split(X, y):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]

    # Load the trained model
    model = GradientBoostingRegressor()

    model.fit(scaler.fit_transform(X_train), y_train)
    
# Streamlit App
st.title('Drug Satisfaction Prediction App')

# User input for prediction
condition_name = st.selectbox('Select Condition:', original_values_condition.unique())
drug_name = st.selectbox('Select Drug:', original_values_drug.unique())
effective = st.slider('Effective:', float(X['Effective'].min()), float(X['Effective'].max()))
ease_of_use = st.slider('Ease of Use:', float(X['EaseOfUse'].min()), float(X['EaseOfUse'].max()))

# Radio buttons for Type and Indication
type_options = ['RX', 'OTC', 'RX/OTC']
selected_type = st.radio('Select Type:', type_options)
indication_options = ['On Label', 'Off Label']
selected_indication = st.radio('Select Indication:', indication_options)

# Convert selected names to encoded values
condition_encoded = label_encoder_condition.transform([condition_name])[0]
drug_encoded = label_encoder_drug.transform([drug_name])[0]

# Create the input_data DataFrame for prediction
input_data = pd.DataFrame({
    'Condition': [condition_encoded],
    'Drug': [drug_encoded],
    'Effective': [effective],
    'EaseOfUse': [ease_of_use],
    'Type_RX': [1 if selected_type == 'RX' else 0],
    'Type_RX/OTC': [1 if selected_type == 'RX/OTC' else 0],
    'Indication_On_Label': [1 if selected_indication == 'On Label' else 0]
})

# Standardize user input
input_data = scaler.transform(input_data)

# Make predictions using the last trained model
prediction = model.predict(input_data)

# Display prediction
st.subheader('Predicted Satisfaction:')
st.write(prediction)
