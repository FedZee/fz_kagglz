''' Kaggle House prices challenge
FedZee sept 2016
'''

''' 0 - preliminary stuff '''

print('Importing libraires ...')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import cross_validation

print('Importing data ...')
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print('Train / test dataframes sizes : ' + str(train_df.shape) + ' / ' + str(test_df.shape))

# features description : numerical or categorical (0 or 1)
features_dict = {"Id": 0, "MSSubClass": 0, "MSZoning": 1, "LotFrontage": 0, "LotArea": 0, "Street": 1, "Alley": 1, \
"LotShape": 1, "LandContour": 1, "Utilities": 1, "LotConfig": 1, "LandSlope": 1, "Neighborhood": 1, "Condition1": 1,  \
"Condition2": 1, "BldgType": 1, "HouseStyle": 1, "OverallQual": 0, "OverallCond": 0, "YearBuilt": 0,  \
"YearRemodAdd": 0, "RoofStyle": 1, "RoofMatl": 1, "Exterior1st": 1, "Exterior2nd": 1, "MasVnrType": 1,  \
"MasVnrArea": 0, "ExterQual": 1, "ExterCond": 1, "Foundation": 1, "BsmtQual": 1, "BsmtCond": 1, "BsmtExposure": 1,  \
"BsmtFinType1": 1, "BsmtFinSF1": 0, "BsmtFinType2": 1, "BsmtFinSF2": 0, "BsmtUnfSF": 0, "TotalBsmtSF": 0,  \
"Heating": 1, "HeatingQC": 1, "CentralAir": 1, "Electrical": 1, "1stFlrSF": 0, "2ndFlrSF": 0, "LowQualFinSF": 0,  \
"GrLivArea": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0, "FullBath": 0, "HalfBath": 0, "BedroomAbvGr": 0,  \
"KitchenAbvGr": 0, "KitchenQual": 1, "TotRmsAbvGrd": 0, "Functional": 1, "Fireplaces": 0, "FireplaceQu": 1, \
"GarageType": 1, "GarageYrBlt": 0, "GarageFinish": 1, "GarageCars": 0, "GarageArea": 0, "GarageQual": 1,  \
"GarageCond": 1, "PavedDrive": 1, "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0, "3SsnPorch": 0,  \
"ScreenPorch": 0, "PoolArea": 0, "PoolQC": 1, "Fence": 1, "MiscFeature": 1, "MiscVal": 0, "MoSold": 0, "YrSold": 0,  \
"SaleType": 1, "SaleCondition": 1, "SalePrice": 0}

''' ------------------------------------------------------------------------
1 - data cleaning & exploration '''

# 1.1 - helper functions
dummy_keys = {}
dummy_calls = 0
def dummy_my_ride(dataframe, features_list): # , feature_keys) :
    ''' changes categorical variables in quick & dirty dummy variables'''
    ''' TBD : assurer cohérence entre les valeurs de conversion des sets train et test
    idée : permettre de passer les clés argument. Si non renseignées, la fonction les génère '''
    # global data
    print('Transforming categorical variables to dummy variables...')
    global dummy_calls
    dummy_calls +=1
    global dummy_keys
    keys = {}
    for feature in features_list :
        dataframe[feature] = dataframe[feature].astype(str)         # a (seemingly) preliminary step to the next formulae
        values = list(enumerate(np.unique(dataframe[feature])))     # determines all values of the feature,
        values_dict = { name : i for i, name in values }            # sets up a dictionary in the form  value : index
        keys[feature] = values_dict                           # saves it for later understanding
        dataframe[feature] = dataframe[feature].apply(lambda x: values_dict[x]).astype(int)     # Converts all features strings to int
        # data.feature = data.feature.map(lambda x: values_dict[x]).astype(int)  '''écriture [feature] ou .feature ? '''
    dummy_keys[dummy_calls] = keys
    print(str(len(keys)) + ' features transformed')

def fill_nans(df):
    ''' Fills NaN with -1 (in order to differentiate it from other values
    TBD: CHECK if there are no -1 values in the data)'''
    for feature in df.columns:
        df[feature] = df[feature].fillna(-1)

# 1.2 - treatment
print('Treating data sets...')
to_be_dummied = [truc for truc in features_dict if features_dict[truc] == 1]
dummy_my_ride(train_df, to_be_dummied)
dummy_my_ride(test_df, to_be_dummied)
# Treating NaNs
'''removing rows - dont do that
train_df = train_df.dropna(axis = 0)
test_df = test_df.dropna(axis = 0)
print('Train / test dataframes sizes after treatment : ' + str(train_df.shape) + ' / ' + str(test_df.shape))'''
fill_nans(train_df)
fill_nans(test_df)

# train_df.describe()

''' ------------------------------------------------------------------------
2 - ML '''
model = RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
# predictors = train_df.drop('SalePrice', axis=1)
# scores = cross_validation.cross_val_score(model, train_df[predictors], train_df["SalePrice"], cv=3)
# print(scores.mean())

# Convert back to a numpy array
train_data = train_df.drop('SalePrice', axis = 1).values
target = train_df['SalePrice']
test_data = test_df.values
# Training
print('Training the machine ...')
model = model.fit( train_data, target )
# Predicting
print('Predicting house prices ...')
predictions = model.predict(test_data).astype(int)
# fonctions helper functions

''' submission '''
submission = pd.DataFrame( {"Id": test_df["Id"], "SalePrice": predictions} )
print('Train prices stats : \n', train_df['SalePrice'].describe())
print('Submission prices stats : \n', submission['SalePrice'].describe())
submission.to_csv("kaggle.csv", index=False)

print('Done.')
