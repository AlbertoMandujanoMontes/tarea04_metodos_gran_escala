"""Script to perform exploratory analysis and train a predictive model"""
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from src.helper_functions import load_process_data


train_data_eda = pd.read_csv("data/raw/train.csv")
test_data_eda = pd.read_csv("data/raw//test.csv")

print("Shape:", train_data_eda.shape)
print("Duplicated data :", train_data_eda.duplicated().sum())

fig, a_x = plt.subplots(figsize=(25,10))
sns.heatmap(data=train_data_eda.isnull(), yticklabels=False, ax=a_x)

fig, a_x = plt.subplots(figsize=(25,10))
sns.countplot(x=train_data_eda['SaleCondition'])
sns.histplot(x=train_data_eda['SaleType'], kde=True, ax=a_x)
sns.violinplot(x=train_data_eda['HouseStyle'], y=train_data_eda['SalePrice'],
               ax=a_x)
sns.scatterplot(x=train_data_eda["Foundation"], y=train_data_eda["SalePrice"],
                palette='deep', ax=a_x)
plt.grid()


# Perform data cleaning process
train_data, test_data = load_process_data(load_from_file=True,
                                          overwrite_file = False)

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
test_ids = test_data['Id']
price = model.predict(test_data)
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": price
})

# Save predictions
submission.to_csv("submission.csv", index=False)
submission.sample(10)
