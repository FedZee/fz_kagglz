'''brouillon
'''

features = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape",  \
"LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",  \
"Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt",  \
"YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",  \
"MasVnrArea", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",  \
"BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",  \
"Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",  \
"GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",  \
"KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "FireplaceQu",  \
"GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual",  \
"GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",  \
"ScreenPorch", "PoolArea", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold",  \
"SaleType", "SaleCondition", "SalePrice"]

'''removing rows - dont do that
train_df = train_df.dropna(axis = 0)
test_df = test_df.dropna(axis = 0)
print('Train / test dataframes sizes after treatment : ' + str(train_df.shape) + ' / ' + str(test_df.shape))'''

''' descriptive stats
print("Exporting desciptive statistics : train_described.csv & test_described.csv ")
train_df.describe().to_csv("train_described.csv", index=False)
test_df.describe().to_csv("test_described.csv", index=False)
'''

''' DEBUG LOG ---------------------------------------------------------------------------

- ValueError: could not convert string to float: 'LwQ'
=> tests : il manque une colonne au describe, pas normal. = 'BsmtFinType2' : mal paramétrée (1 et non 0)

- ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
=> dropna

- ValueError: Number of features of the model must  match the input. Model n_features is 77 and  input n_features is 69
=> def features_fromA_not_in_B(dfA, dfB):
non
=> dropna axis = 0 et non 1 ! pour dropper les lignes et non les colonnes

debug cross_validation
ValueError: Must pass DataFrame with boolean values only

'''

''' PREDICTIONS LOG

1-
date: 18 sept 2016
model : model = RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
score : 0.14821
Submission prices stats :  count      1459.000000
mean     178613.169294
std       72665.108054
min       59668.000000
25%      129218.500000
50%      157409.000000
75%      210280.000000
max      535333.000000
'''
