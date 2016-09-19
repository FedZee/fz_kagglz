''' Kaggle House prices challenge
FedZee sept 2016
'''

''' 0 - preliminary stuff '''

print('Importing libraires ...')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
categorical_feat = [truc for truc in features_dict if features_dict[truc] == 1]
dummy_my_ride(train_df, test_df, categorical_feat)
# Treating NaNs
fill_nans(train_df)
fill_nans(test_df)

''' ------------------------------------------------------------------------
2 - ML '''
''' settings '''
# Tidying features
size_feat = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', \
'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', \
'ScreenPorch', 'PoolArea']
categ_n_size = categorical_feat + size_feat
numerical_feat = [feat for feat in features_dict if features_dict[feat] == 0]
# settings algos
algorithms = [ \
(RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2),
categ_n_size), \
(LinearRegression(), numerical_feat) ]
weight = [1, 1]

''' program '''
# Convert back to a numpy array
train_data = train_df.drop('SalePrice', axis = 1).values
target = train_df['SalePrice']
test_data = test_df.values


full_predictions = []
for algo, predictors in algorithms:
    # Training
    print('Training the machine ...')
    model = model.fit( train_data, target )
    # Predicting
    print('Predicting house prices ...')
    predictions = model.predict(test_data).astype(int)   # <class 'numpy.ndarray'> ([:,1] in tuto?)
    full_predictions.append(predictions)

predictions = (full_predictions[0]*weight[0] \
+ full_predictions[1]*weight[1] + full_predictions[2]*weight[2]) / weight[3]

''' submission '''
submission = pd.DataFrame( {"Id": test_df["Id"], "SalePrice": predictions} )
print('Train prices stats : \n', train_df['SalePrice'].describe())
print('Submission prices stats : \n', submission['SalePrice'].describe())
submission.to_csv("kaggle.csv", index=False)

print('Done.')

''' TODO ---------------------------------------------------------------------------
OK - vérifier les valeurs -1
OK - Vérifier / corriger (systématiser) la cohérence entre les 2 tablaux de correspondance des dummy, dans train & test
- comprendre score kaggle
- faire des cv test auto (débugguer :check docu?) sur plusieurs types de modeles et paramétrages
- étudier les variables à la main. Les comprendre.
    savoir faire une projection à la main
    plots de partout. reprendre les plots de velib
- nouveautés à apprendre
    - tester XGBoost library (pour RF et GBM)
    - utiliser Kernels / community learning
    - lire docu sur les ensembles, afin d'en extraire les meilleures pratiques
    + check autres trucs d'og 
    - data analytics style
- faire une extraction des features lineaires les plus importants
- faire d'autres analyses d'importance ? Cf. article de bidule sur datascience.net
- scores : générer score kaggle et en extraire d'autres pour compréhension
- étudier le overkill analytics
- Feature engineering :
    - regrouper des features ?
        - surfaces
        - nombre de SdB, etc.
    - séparer les misc features ?
    - remplacer year sold par le prix moyen au metre carré de l'année considérée * surface
