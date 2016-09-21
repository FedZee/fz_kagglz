''' Kagl Houses challenge
FedZee sept 2016
'''

''' 0 - preliminary stuff '''

print('Importing libraires ...')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics
import math
#import time

print('Importing data ...')
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print('Train / test dataframes sizes : ' + str(train_df.shape) + ' / ' + str(test_df.shape))

''' ------------------------------------------------------------------------
1 - data cleaning & exploration '''

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
"SaleType": 1, "SaleCondition": 1}

# 1.1 - helper functions

# Changing categorical values to integers (dummy variables)
def categ_values(df, features):
    '''Lists all the different values taken by each given cat. feature, and indexes them'''
    keys = {}
    for feature in features :
        df[feature] = df[feature].astype(str)         # a (seemingly) preliminary step to the next formulae
        values = list(enumerate(np.unique(df[feature])))     # determines all values of the feature,
        values_dict = { name : i for i, name in values }            # sets up a dictionary in the form  value : index
        keys[feature] = values_dict
    return keys
def generate_dummy_keys(train, test, features):
    ''' generates consistent dummy keys between train and test sets .
    Saves the keys and the differences between test and train's values'''
    global dummy_values, dummy_differences
    # load the values and indexes (dummy keys) for all the features, initialize consistent values
    dummy_values = {'train': categ_values(train, features), 'test': \
    categ_values(test, features), 'test_consistent': {} }
    for feature in features :
        # get the feature's values and their dummy keys
        train_values, test_values = dummy_values['train'][feature], dummy_values['test'][feature]
        # make the test set's dummies consistent with the train set's
        test_values_full = train_values       # take the train set's dummy values as a basis for test set
        train_only_values = [value for value in train_values if value not in test_values]
        test_only_values = [value for value in test_values if value not in train_values]
        dummy_differences[feature] = {'train only': train_only_values, 'test only': test_only_values}
        for value in test_only_values:   # reassign dummy keys
            test_values_full[value] = test_values[value] + len(train_values)  # gives a new int value that does not exist in train values
        dummy_values['test_consistent'][feature] = test_values_full
def dummy_my_ride(train, test, features): # , feature_keys) :
    ''' Changes categorical variables in quick dummy variables,
    for both train and test set.'''
    generate_dummy_keys(train, test, features)  # generate keys
    global dummy_values                         # and access them
    print('Transforming categorical variables to dummy variables...')
    for feature in features :
        # Load the keys
        train_values, test_values_full = dummy_values['train'][feature], dummy_values['test_consistent'][feature]
        # Convert all features strings to consistent dummy values integers
        train[feature] = train[feature].apply(lambda x: train_values[x]).astype(int)
        test[feature] = test[feature].apply(lambda x: test_values_full[x]).astype(int) # other possibility : using .feature ? .map method ?
    print(str(len(features)) + ' features transformed. Dummy values and differences saved.')
# Dealing with NaNs
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
2 - ML settings - set the models here '''

# Classifying features
target = 'SalePrice' # Extract target
all_feat = list(features_dict.keys())
size_feat = ['LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', \
'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', \
'ScreenPorch', 'PoolArea']
categ_n_size = categorical_feat + size_feat
numerical_feat = [feat for feat in features_dict if features_dict[feat] == 0]

# settings algorithms
rf1 = RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
rf_overfit = RandomForestRegressor(random_state=1, n_estimators=150) # default : min_samples_split=2, min_samples_leaf=1
gbm = GradientBoostingRegressor(random_state=1, n_estimators=25, max_depth=3)
# these will be the chosen algorithms (& features set) for test, and how you blend them
algorithms = [ (rf1, categ_n_size), (rf1, all_feat), (rf_overfit, all_feat), (gbm, all_feat)]
# (LinearRegression(), numerical_feat) ] # NB : bug (Cf. bug log)
weights = [0, 1, 0, 1] # if number of weights does not match the algorithms', will be filled with 0's

''' ------------------------------------------------------------------------
3 - ML program '''

# 3.1 - Helper functions
def rmsle(y_true, y_pred): # arguments : arrays or lists ; returns : score
    ''' Computes the RMSE on logarithmic error = RMSE(logx-logy)
    Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the
    logarithm of the predicted value and the logarithm of the observed sales price.'''
    # y_pred = fix_negatives(y_pred, y_true) # did not help to fix linear regr° bug
    y_true_log = np.log(y_true + 1)
    y_pred_log = np.log(y_pred + 1)
    count = 0
    try :
        return np.sqrt( metrics.mean_squared_error(y_true_log, y_pred_log) )
    except Exception as error :
        print('Error in RMSLE function : ' + str(type(error)) + ' ; ' + str(error))
    return 0
def rmse(y_true, y_pred):
    ''' RMSE. Somewhat more insightful metric. '''
    return np.sqrt( metrics.mean_squared_error(y_true, y_pred) )
def report_scores(y_pred, y_true) : # input : y_pred : array ; y_true : Series
    '''Reports scores ...'''
    try :
        print('RMSLE : ' + str(rmsle(y_pred, y_true.values)))
    except Exception as error :
        print('Unexpected error for RMSLE : ' + str(type(error)) + ' ; ' + str(error))
    try :
        print('RMSE : ' + str(rmse(y_pred, y_true.values)))
    except Exception as error :
        print('Unexpected error for RMSE : ' + str(type(error)) + ' ; ' + + str(error))
def train_one_model(alg, predictors_labels, dataframe, target) :
    ''' Trains & returns a model, and reports about it'''
    kf = KFold(dataframe.shape[0], n_folds=3, random_state=1)
    predictions = [] # liste dans laquelle on va stocker les arrays de predictions
    for train, test in kf:
        # Setting predictors and target.
        train_predictors = (dataframe[predictors_labels].iloc[train,:])
        train_target = dataframe[target].iloc[train]
        # Training the algorithm
        alg.fit(train_predictors, train_target)
        # Make predictions on the test fold
        test_predictions = alg.predict(dataframe[predictors_labels].iloc[test,:])
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0) # retourne un array
    print(type(alg))
    print('Number of features used : ' + str(len(predictors_labels)))
    report_scores(predictions, dataframe[target])
    return alg, predictions
# train_one_model(rf1, train_df, all_feat, target)
def train_multiple_models(algorithms, dataframe, target) :
    ''' tests models, reports results (RMSLE),
    returns the models ready for use + predictions for analysis '''
    trained_models = []
    all_predictions = []
    count = 0
    for alg, predictors in algorithms :
        count +=1
        print('Model n°' + str(count))
        alg_trained, predictions = train_one_model(alg, predictors, dataframe, target)
        trained_models.append(alg_trained)
        all_predictions.append(predictions)
    return trained_models, all_predictions
def apply_models(trained_models, weights, test_df) :
    # Predict
    full_predictions = []
    count = 0
    for alg in trained_models :
        count += 1
        print('Applying model n°' + str(count))
        prediction = alg.predict(test_df)
        full_predictions.append(prediction)
    return full_predictions
def blend(full_p, weights) :
    ''' Blends predictions: computes weighted average of the regression '''
    sigma = 0
    # ensuring consistency of inputs
    weights = weights[:len(full_p)]
    weights[len(weights) : len(full_p)] = [0 for i in range(len(full_p) - len(weights))]
    # computing the blending
    for number in range(len(full_p)) :
        sigma += (full_p[number] * weights[number])
    print('Models blended with the following weights :')
    print(weights)
    return sigma / sum(weights)
def report_stats(trained_models, all_predictions):
    for i in range(len(all_predictions)) :
        print('- Model n°' + str(i + 1))
        print(type(trained_models[i]))
        print('Model prices stats :')
        print(pd.Series(all_predictions[i]).describe())

# 3.2 Executing ML
# Train & report statitics on results
print('\nTraining & testing models ...')
trained_models, all_predictions = train_multiple_models(algorithms, train_df, target)
print('\nResults :')
print('Actual prices stats : \n', train_df['SalePrice'].describe())
report_stats(trained_models, all_predictions)
# Make predictions
print('\nMaking predictions ...')
full_predictions = apply_models(trained_models, weights, test_df)
predictions = blend(full_predictions, weights)
# Submit
print('\nExporting submission file')
submission = pd.DataFrame( {"Id": test_df["Id"], "SalePrice": predictions} )
print('Submission prices stats : \n', submission['SalePrice'].describe())
submission.to_csv("kaggle.csv", index=False)
print('Done.')


''' TODO ---------------------------------------------------------------------------
OK - vérifier les valeurs -1
OK - Vérifier / corriger (systématiser) la cohérence entre les 2 tablaux de correspondance des dummy, dans train & test
OK - permettre la saisie facilitée d'autants de modèles que voulu + blending auto
OK - rajouter score kaggle
    - rajouter Kfold testing sur mon set
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
'''

''' BUG LOG
- Bug de la régression linéaire
    - bug 1 : RMSLE
<class 'ValueError'>
Input contains NaN, infinity or a value too large for dtype('float64').
    - bug 2 : application de l'algo sur le test set
--> 184         prediction = alg.predict(test_df)
ValueError: shapes (1459,80) and (37,) not aligned: 80 (dim 1) != 37 (dim 0)
'''
