''' Kaggle House prices challenge
FedZee sept 2016
'''

''' 0 - preliminary stuff '''

print('Importing libraires ...')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import time

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


def categ_values(df, features):
    keys = {}
    for feature in features :
        df[feature] = df[feature].astype(str)         # a (seemingly) preliminary step to the next formulae
        values = list(enumerate(np.unique(df[feature])))     # determines all values of the feature,
        values_dict = { name : i for i, name in values }            # sets up a dictionary in the form  value : index
        keys[feature] = values_dict
    return keys

def dummy_my_ride(train, test, features): # , feature_keys) :
    ''' Changes categorical variables in quick & dirty dummy variables.
    Saves the dummy keys and the differences between test and train's values'''
    print('Transforming categorical variables to dummy variables...')
    global dummy_values, dummy_differences
    dummy_values = {'train': categ_values(train, features), 'test': categ_values(test, features)}
    for feature in features :
        # get the values and their dummy keys
        train_values, test_values = dummy_values['train'][feature], dummy_values['test'][feature]
        # make the test set's dummies consistent with the train set's
        test_values_full = train_values                   # uses the train set's dummy values
        train_only_values = [value for value in train_values if value not in test_values]
        test_only_values = [value for value in test_values if value not in train_values]
        dummy_differences[feature] = {'train only': train_only_values, 'test only': test_only_values}
        for value in test_only_values:
            test_values_full[value] = test_values[value] + len(train_values)  # gives a new int value that does not exist in train values
        # Convert all features strings to int
        train[feature] = train[feature].apply(lambda x: train_values[x]).astype(int)
        test[feature] = test[feature].apply(lambda x: test_values_full[x]).astype(int) # other possibility : using .feature ? .map method ?
    print(str(len(features)) + ' features transformed. Dummy values and differences saved.')

def fill_nans(df):
    ''' Fills NaN with -1 (in order to differentiate it from other values. Check done : no negative values in dataset '''
    for feature in df.columns:
        df[feature] = df[feature].fillna(-1)

# 1.2 - treatment
print('Treating data sets...')
dummy_values = {}
dummy_differences = {}
to_be_dummied = [truc for truc in features_dict if features_dict[truc] == 1]
dummy_my_ride(train_df, test_df, to_be_dummied)
# Treating NaNs
'''removing rows - dont do that
train_df = train_df.dropna(axis = 0)
test_df = test_df.dropna(axis = 0)
print('Train / test dataframes sizes after treatment : ' + str(train_df.shape) + ' / ' + str(test_df.shape))'''
fill_nans(train_df)
fill_nans(test_df)
print("Exporting desciptive statistics : train_described.csv & test_described.csv ")
train_df.describe().to_csv("train_described.csv", index=False)
test_df.describe().to_csv("test_described.csv", index=False)

''' ------------------------------------------------------------------------
2 - ML '''
model = RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

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
