# Vaccine Hesitancy ML Project

## 1. Overview

This repository implements a machine learning pipeline to analyse **COVID-19 vaccine hesitancy** in the United States. Using survey data (two waves) from Harvard Dataverse, we classify respondents as **vaccine hesitant** (`vaccine_hesitant`) or not, based on:

- socio-demographic characteristics (age, gender, race, income, college education, etc.)
- political attitudes and trust in government and media
- trust in science and perceived risk of COVID-19
- health conditions and previous infection history

This could help decisionmakers in case of a new epidemic, to estimate the amount of people that can be vaccinated, which is a crucial feature of handling a situation like that.

The final goal is to:
- build predictive models (e.g. Lasso, Ridge, Logistic Regression, Tree-based models), and  
- interpret which features most strongly predict vaccine hesitancy.

---

## 2. Data

- **Original dataset**: Harvard Dataverse (COVID vaccine hesitancy survey; data/vh_data14.csv)  
- **Wave 1**: no vaccine-hesitancy label → used later for *prediction only*  
- **Wave 2**: includes `vaccine_hesitant` → used as *training dataset*
- Find detailed variable descriptions in the file "variable_descriptions.docx".
- Link to the original file: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FGJVWYF&fbclid=IwY2xjawORHtpleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeFA8hkI5DVFFs3e4UinyZszS9x4FfHL5gNE-EBJfltBWlE_9Jjxoyyeh3FMs_aem_L2iGJBJikODnIj3Ct4ysTQ

## 3 Setup and Exectuion

### 3.1 Download repository, unzip it, and enter the folder vax-hesitancy-ml-main in a terminal window
    cd Downloads/vax-hesitancy-ml-main

### 3.2 Create virtual environment
    python -m venv .venv

    or 

    python3 -m venv .venv

### 3.3 Activate the environment
    Windows:
     .venv\Scripts\activate
    macOS / Linux:
     source .venv/bin/activate

### 3.4 Install all required packages
    pip install -r requirements.txt

### 3.6 Run pipeline (arguments --data_path dataset/vh_data14.csv and --seed 101 are optional)
    python main.py

## Expected runtime can vary between 5 to 10 minutes based on hardware specifications.




