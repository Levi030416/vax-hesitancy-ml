# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Countplot of vaccine hesitancy
sns.countplot(x='vaccine_hesitant', data=vh_wave2)
plt.title('Distribution of Vaccine Hesitancy (Wave 2)')
plt.show()

## Boxplots for numeric variables
numeric_cols = [
    'individual_responsibility', 'trust_science_community', 'trust_science_polmotives',
    'trust_science_politicians', 'trust_science_media', 'trust_media',
    'trust_gov_nat', 'trust_gov_state', 'trust_gov_local',
    'perceived_personal_riskq297_4', 'perceived_network_risk',
    'doctor_comfort', 'fear_needles', 'income', 'county_density',
    'age','psindex', 'nsindex', 'pandemic_impact_personal', 'pandemic_impact_network',
    'vaccine_trust']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='vaccine_hesitant', y=col, data=vh_wave2)
    plt.title(f'{col} vs Vaccine Hesitancy')
    plt.show()
    
## Plots for categorical variables
categorical_cols = [
    'male', 'college', 'evangelical', 'infected_personal', 'infected_network',
    'president_approval', 'trump_approval_retrospective', 'condition_pregnant',
    'condition_asthma', 'condition_lung', 'condition_diabetes', 'condition_immune',
    'condition_obesity', 'condition_heart', 'condition_organ', 'race', 'party_id']

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='vaccine_hesitant', data=vh_wave2)
    plt.title(f'{col} by Vaccine Hesitancy')
    plt.show()
    
    
## Correlation matrix for numeric variables
plt.figure(figsize=(12,10))
sns.heatmap(vh_wave2[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features (Wave 2)')
plt.show()

## Correlation with target variable
cor_target = vh_wave2[numeric_cols + ['vaccine_hesitant']].corr()['vaccine_hesitant'].sort_values(ascending=False)
print("Correlation with vaccine_hesitant:\n", cor_target)