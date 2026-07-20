"""
Trains the laptop price prediction model, replicating the feature engineering
performed in notebooks_laptop-price-predictor.ipynb, and saves:
  - df.pkl   -> the cleaned dataframe (used by the app to populate dropdowns)
  - pipe.pkl -> the trained sklearn Pipeline (preprocessing + RandomForestRegressor)

Note: the original notebook's final model was a StackingRegressor that included
XGBoost. XGBoost isn't available in this environment, so RandomForestRegressor
is used instead — it was one of the strongest individual models in the notebook
(R2 ~0.88) and needs no extra dependencies beyond scikit-learn. If you have
xgboost installed locally, feel free to swap back to the Stacking/XGB model —
the app code doesn't need to change, since it just loads pipe.pkl.
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv('laptop_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# ---------------------------------------------------------------------------
# 2. Ram / Weight cleanup
# ---------------------------------------------------------------------------
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# ---------------------------------------------------------------------------
# 3. Screen resolution -> Touchscreen, IPS, PPI
# ---------------------------------------------------------------------------
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0]
df['Y_res'] = new[1]

df['X_res'] = df['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

df['ppi'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5 / df['Inches']).astype('float')

df.drop(columns=['ScreenResolution'], inplace=True)
df.drop(columns=['Inches', 'X_res', 'Y_res'], inplace=True)

# ---------------------------------------------------------------------------
# 4. CPU brand
# ---------------------------------------------------------------------------
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))


def fetch_processor(text):
    if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return text
    if text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    return 'AMD Processor'


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# ---------------------------------------------------------------------------
# 5. Memory -> HDD / SSD / Hybrid / Flash storage totals
# ---------------------------------------------------------------------------
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)

new = df['Memory'].str.split('+', n=1, expand=True)
df['first'] = new[0].str.strip()
df['second'] = new[1].fillna('0').astype(str).str.strip()

df['Layer1HDD'] = df['first'].str.contains('HDD').astype(int)
df['Layer1SSD'] = df['first'].str.contains('SSD').astype(int)
df['Layer1Hybrid'] = df['first'].str.contains('Hybrid').astype(int)
df['Layer1Flash_Storage'] = df['first'].str.contains('Flash Storage').astype(int)
df['first'] = df['first'].str.replace(r'\D', '', regex=True).astype(int)

df['Layer2HDD'] = df['second'].str.contains('HDD').astype(int)
df['Layer2SSD'] = df['second'].str.contains('SSD').astype(int)
df['Layer2Hybrid'] = df['second'].str.contains('Hybrid').astype(int)
df['Layer2Flash_Storage'] = df['second'].str.contains('Flash Storage').astype(int)
df['second'] = df['second'].str.replace(r'\D', '', regex=True).astype(int)

df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid'])
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage'])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
                  'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'], inplace=True)
df.drop(columns=['Memory'], inplace=True)
df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True, errors='ignore')

# ---------------------------------------------------------------------------
# 6. GPU brand
# ---------------------------------------------------------------------------
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)

# ---------------------------------------------------------------------------
# 7. Operating system
# ---------------------------------------------------------------------------
def cat_os(inp):
    if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    if inp in ('macOS', 'Mac OS X'):
        return 'Mac'
    return 'Others/No OS/Linux'


df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

# ---------------------------------------------------------------------------
# 8. Train / test split
# ---------------------------------------------------------------------------
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Column indices of categorical text columns to one-hot encode
# (Company, TypeName, Cpu brand, Gpu brand, os) -> match notebook indices [0,1,7,10,11]
cat_cols_idx = [X.columns.get_loc(c) for c in ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']]

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), cat_cols_idx)
], remainder='passthrough')

step2 = RandomForestRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15,
)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

# ---------------------------------------------------------------------------
# 9. Export
# ---------------------------------------------------------------------------
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print('Saved df.pkl and pipe.pkl')
