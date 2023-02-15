"""Script to perform exploratory analysis and train a predictive model"""
import logging
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from src.helper_functions import load_process_data


parser = argparse.ArgumentParser(
    prog="script",
    usage = "Indicate if you want to load data from raw or clean",
    description= "Train the model"
)

parser.add_argument('source', type=str)
parser.add_argument('-o', action='store_true', default=False)


args = parser.parse_args()



logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s')

#This code is meant to be and EDA, no suitable for production,
# it is left commented in case we need to run the EDA again in te future
#train_data_eda = pd.read_csv("data/raw/train.csv")
#test_data_eda = pd.read_csv("data/raw//test.csv")
#
#print("Shape:", train_data_eda.shape)
#print("Duplicated data :", train_data_eda.duplicated().sum())

#fig, a_x = plt.subplots(figsize=(25,10))
#sns.heatmap(data=train_data_eda.isnull(), yticklabels=False, ax=a_x)

#fig, a_x = plt.subplots(figsize=(25,10))
#sns.countplot(x=train_data_eda['SaleCondition'])
#sns.histplot(x=train_data_eda['SaleType'], kde=True, ax=a_x)
#sns.violinplot(x=train_data_eda['HouseStyle'], y=train_data_eda['SalePrice'],
#               ax=a_x)
#sns.scatterplot(x=train_data_eda["Foundation"], y=train_data_eda["SalePrice"],
#                palette='deep', ax=a_x)
#plt.grid()


# Perform data cleaning process

if args.source == 'raw':
    load_flag = False
elif args.source == 'clean':
    load_flag = True
else:
    load_flag = True
    logging.warning("Source not found, using clean as default")

train_data, test_data = load_process_data(load_from_file=load_flag,
                                          overwrite_file = args.o)

if train_data.shape[0] == 0:
    logging.warning("Train set is empty")

if test_data.shape[0] == 0:
    logging.warning("test set is empty")

y = train_data['SalePrice']
X = train_data.drop(['SalePrice'], axis=1)

#Train the model
candidate_max_leaf_nodes = [250]
for node in candidate_max_leaf_nodes:
    model = RandomForestRegressor(max_leaf_nodes=node,)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10)
    print(score.mean())


# Make predictions
test_ids = test_data.index
price = model.predict(test_data)
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": price
})

# Save predictions
submission.to_csv("submission.csv", index=False)
submission.sample(10)
