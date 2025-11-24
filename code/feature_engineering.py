# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

y.info()
X_test.info()
X_train.info()

## Feature engineering

## Data types

# making dummies out of president_approval and trump_approval_retrospective
X_train['president_approval'] = X_train['president_approval'].map({'Yes': 1, 'No': 0})
X_train['trump_approval_retrospective'] = X_train['trump_approval_retrospective'].map({'Yes': 1, 'No': 0})

X_test['president_approval'] = X_test['president_approval'].map({'Yes': 1, 'No': 0})
X_test['trump_approval_retrospective'] = X_test['trump_approval_retrospective'].map({'Yes': 1, 'No': 0})


## Converting categorical columns
categorical_cols = [
    'male', 'college', 'evangelical', 'infected_personal', 'infected_network',
    'president_approval', 'trump_approval_retrospective',
    'condition_pregnant', 'condition_asthma', 'condition_lung', 'condition_diabetes',
    'condition_immune', 'condition_obesity', 'condition_heart', 'condition_organ', 'race', 'party_id']

for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
  
## Converting numeric columns
numeric_cols = [
    'individual_responsibility', 'trust_science_community', 'trust_science_polmotives',
    'trust_science_politicians', 'trust_science_media', 'trust_media',
    'trust_gov_nat', 'trust_gov_state', 'trust_gov_local',
    'perceived_personal_riskq297_4', 'perceived_network_risk',
    'doctor_comfort', 'fear_needles', 'income', 'county_density',
    'age','psindex', 'nsindex', 'pandemic_impact_personal', 'pandemic_impact_network',
    'vaccine_trust']

for col in numeric_cols:
    X_train[col] = X_train[col].astype('float')
    X_test[col] = X_test[col].astype('float')

# Making dummies out of categorical variables that are not dummies
X_train = pd.get_dummies(X_train, columns=['race', 'party_id'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['race', 'party_id'], drop_first=True)




## Imputing missing data
median_imputer = SimpleImputer(strategy='median')
X_train[numeric_cols] = median_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = median_imputer.transform(X_test[numeric_cols])

categorical_cols = [
    'male', 'college', 'evangelical', 'infected_personal', 'infected_network',
    'president_approval', 'trump_approval_retrospective',
    'condition_pregnant', 'condition_asthma', 'condition_lung', 'condition_diabetes',
    'condition_immune', 'condition_obesity', 'condition_heart', 'condition_organ', 'race_2', 'race_3', 'race_4', 
    'party_id_Independent', 'party_id_Libertarian', 'party_id_Other party', 'party_id_Republican']

mode_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = mode_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = mode_imputer.transform(X_test[categorical_cols])

for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

X_train.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)


## Separate Standardized dataset for Lasso and Ridge regressions
X_train_standardized = X_train.copy()
X_test_standardized = X_test.copy()
scaler = StandardScaler()
X_train_standardized[numeric_cols] = scaler.fit_transform(X_train_standardized[numeric_cols])
X_test_standardized[numeric_cols] = scaler.transform(X_test_standardized[numeric_cols])

X_train_standardized.to_csv("X_train_standardized.csv", index=False)
X_test_standardized.to_csv("X_test_standardized.csv", index=False)


