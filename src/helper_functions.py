"""This library contains helper functions to perform the analysis"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd



def fill_all_missing_values(data):
    """
    Fill missing values of an entire dataframe,
    filling with mean if the column is numeric and mode in any other case
    """
    for col in data.columns:
        if data[col].dtype in ('float64', 'int64'):
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

def encode_catagorical_columns(train, test, level_col):
    """
    Function that encondes castegorical variables
    :param train: pandas object containing train set
    :param test: pandas object containing test set
    :param Level_col: column to be encoded
    :return: exit code
    """
    encoder = LabelEncoder()
    for col in level_col:
        train[col] = encoder.fit_transform(train[col])
        test[col]  = encoder.transform(test[col])
    return 0

def apply_ordinal_encoder(columns, levels, train_par, test_par):
    """
    Function to aply ordinal encoder to several columns
    :param column: column to be transformed
    :param levels: list of ordered leves to be used
    :param train_par train data set
    :param test_par test data set
    :return: Exit code
    """
    o_e = OrdinalEncoder(categories=levels)
    for column in columns:
        train_par[column] = o_e.fit_transform(train_par[[column]])
        test_par[column] = o_e.transform(test_par[[column]])
    return 0

def load_process_data(load_from_file, overwrite_file):
    """
    Function that reads raw data and transform it
    :param load_from_file: Flag to re run the cleaning process or read from a
    a feahter file
    :param overwrite_file: Flag to override the current feather file
    :return: train and test dataframes.
    """
    if load_from_file:
        train_data = pd.read_feather('data/clean/train_clean.feather')
        test_data = pd.read_feather('data/clean/test_clean.feather')
    else:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw//test.csv")

        # Fill missing data
        train_data['FireplaceQu'].fillna("No", inplace=True)
        train_data['BsmtQual'].fillna("No", inplace=True)
        train_data['BsmtCond'].fillna("No", inplace=True)
        train_data['BsmtFinType1'].fillna("No", inplace=True)
        train_data['BsmtFinType2'].fillna("No", inplace=True)
        train_data['BsmtFinType2'].fillna("None", inplace=True)

        fill_all_missing_values(train_data)
        fill_all_missing_values(test_data)

        # Drop unwanted data
        drop_col = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold',
                    'YrSold', 'MSSubClass', 'GarageType', 'GarageArea',
                    'GarageYrBlt',
                    'GarageFinish', 'YearRemodAdd', 'LandSlope', 'BsmtUnfSF',
                    'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1',
                    'Condition2', 'Heating', 'Exterior1st', 'Exterior2nd',
                    'HouseStyle', 'LotShape', 'LandContour', 'LotConfig',
                    'Functional', 'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu',
                    'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
                   ]

        train_data.drop(drop_col, axis=1, inplace=True)
        test_data.drop(drop_col, axis=1, inplace=True)

        # Preprocessing

        apply_ordinal_encoder(['BsmtQual', 'BsmtCond'],
                              [['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']],
                              train_data, test_data)
        apply_ordinal_encoder(['ExterQual', 'ExterCond','KitchenQual'],
                              [['Po', 'Fa', 'TA', 'Gd', 'Ex']],
                             train_data, test_data)
        apply_ordinal_encoder(['Electrical'],
                              [['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']] ,
                             train_data, test_data)
        apply_ordinal_encoder(['PavedDrive'],
                              [['N', 'P', 'Y']] ,
                              train_data, test_data)
        apply_ordinal_encoder(['BsmtFinType1','BsmtFinType2'],
                              [['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']],
                              train_data, test_data)
        apply_ordinal_encoder(['Utilities'],
                              [['ELO', 'NoSeWa', 'NoSewr', 'AllPub']],
                              train_data, test_data)
        apply_ordinal_encoder(['MSZoning'],
                              [['C (all)', 'RH', 'RM', 'RL', 'FV']],
                              train_data, test_data)
        apply_ordinal_encoder(['Foundation'],
                              [['Slab', 'BrkTil', 'Stone', 'CBlock', 'Wood', 'PConc']],
                              train_data, test_data)
        apply_ordinal_encoder(['Neighborhood'],
                              [['MeadowV', 'IDOTRR', 'BrDale', 'Edwards', 'BrkSide',
                                'OldTown', 'NAmes', 'Sawyer', 'Mitchel', 'NPkVill',
                                'SWISU', 'Blueste', 'SawyerW', 'NWAmes', 'Gilbert',
                                'Blmngtn', 'ClearCr', 'Crawfor', 'CollgCr',
                                'Veenker', 'Timber', 'Somerst', 'NoRidge', 'StoneBr',
                                'NridgHt']],
                              train_data, test_data)
        apply_ordinal_encoder(['MasVnrType'],[['None', 'BrkCmn', 'BrkFace', 'Stone']],
                              train_data, test_data)
        apply_ordinal_encoder(['SaleCondition'],
                              [['AdjLand', 'Abnorml','Alloca', 'Family',
                                'Normal', 'Partial']],
                              train_data, test_data)
        apply_ordinal_encoder(['RoofStyle'],
                              [['Gambrel', 'Gable','Hip', 'Mansard',  'Flat', 'Shed']],
                              train_data, test_data)

        apply_ordinal_encoder(['RoofMatl'],
                              [['ClyTile', 'CompShg', 'Roll','Metal', 'Tar&Grv',
                                'Membran', 'WdShake', 'WdShngl']],
                              train_data, test_data)


        # Encode categorical columns
        encode_catagorical_columns(train_data, test_data,
                                   ['Street' ,'BldgType', 'SaleType', 'CentralAir'])

        # Feature engineering
        train_data['BsmtRating'] = train_data['BsmtCond'] * train_data['BsmtQual']
        train_data['ExterRating'] = train_data['ExterCond'] * train_data['ExterQual']
        train_data['BsmtFinTypeRating'] = \
            train_data['BsmtFinType1'] * train_data['BsmtFinType2']

        train_data['BsmtBath'] = \
            train_data['BsmtFullBath'] + train_data['BsmtHalfBath']
        train_data['Bath'] = train_data['FullBath'] + train_data['HalfBath']
        train_data['PorchArea'] = \
            train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + \
            train_data['3SsnPorch'] + train_data['ScreenPorch']

        test_data['BsmtRating'] = test_data['BsmtCond'] * test_data['BsmtQual']
        test_data['ExterRating'] = test_data['ExterCond'] * test_data['ExterQual']
        test_data['BsmtFinTypeRating'] = \
            test_data['BsmtFinType1'] * test_data['BsmtFinType2']

        test_data['BsmtBath'] = test_data['BsmtFullBath'] + test_data['BsmtHalfBath']
        test_data['Bath'] = test_data['FullBath'] + test_data['HalfBath']
        test_data['PorchArea'] = \
            test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + \
            test_data['3SsnPorch'] + test_data['ScreenPorch']

        # Drop unnecesary columns
        drop_col = ['OverallQual',
                    'ExterCond', 'ExterQual',
                    'BsmtCond', 'BsmtQual',
                    'BsmtFinType1', 'BsmtFinType2',
                    'HeatingQC',
                    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                    'BsmtFullBath', 'BsmtHalfBath',
                    'FullBath', 'HalfBath',
                   ]

        train_data.drop(drop_col, axis=1, inplace=True)
        test_data.drop(drop_col, axis=1, inplace=True)

        if overwrite_file:
            train_data.to_feather('data/clean/train_clean.feather')
            test_data.to_feather('data/clean/test_clean.feather')

    return train_data, test_data
