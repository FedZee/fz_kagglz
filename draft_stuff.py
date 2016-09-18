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

''' DEBUG LOG

- ValueError: could not convert string to float: 'LwQ'
=> tests : il manque une colonne au describe, pas normal. = 'BsmtFinType2' : mal paramétrée (1 et non 0)

- ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
=> dropna

- ValueError: Number of features of the model must  match the input. Model n_features is 77 and  input n_features is 69
=> def features_fromA_not_in_B(dfA, dfB):
non
=> dropna axis = 0 et non 1 ! pour dropper les lignes et non les colonnes

'''

''' PREDICTIONS LOG

1-
date: 18 sept
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

''' TODO
vérifier les valeurs -1
Vérifier / corriger (systématiser) la cohérence entre les 2 tablaux de correspondance des dummy, dans train & test
comprendre score kaggle
faire des cv test auto sur plusieurs types de modeles et paramétrages
faire une extraction des features lineaires les plus importants
faire d'autres analyses d'importance ? Cf. article de bidule sur datascience.net
scores : générer score kaggle et en extraire d'autres pour compréhension
étudier le overkill analytics
étudier les variables à la main. Les comprendre.
    plots de partout. reprendre les plots de velib
