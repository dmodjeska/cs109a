# Predicting Loan Outcomes using Machine Learning
## by David Modjeska and Andrew Greene
<br>

### Course Project
### CS109a: Introduction to Data Science
### Harvard University, Autumn 2016
<br>

### Description

Peer-to-peer loans issued by the Lending Club (LC) require investors to analyze substantial data before funding. The research literature shows that prediction with this data is challenging. We combine LC data with external economic indicators to establish a foundation for modeling. Using a consistent framework, we model with a range of machine learning techniques, including logistic regression, random forests, and stacking. We supplement these models with NLP techniques applied to the free-text description field that is present in many loan applications. Results show the importance of clarifying the actual investor problem, limitations in the published LC data, and the financial leverage that can be achieved from small statistical gains.

### Main Code Files
* DKM_Project_Clean_and_Process.ipynb: clean and pre-process the raw data
* Data_Loading.ipynb and data_loading.py: notebook and text forms of code to load and split the preprocessed data
* amg_explore.ipynb: generate exploratory graphs and tables
* Modelling_Framework.ipynb and modelling_framework.py: notebook and text forms of code to fit, score, and graph individual models
* model_performance.py: render the final modelling results
* Main.ipynb: model the loaded data using the modelling framework
* emergency.py: A standalone python script to re-run everything in parallel at the last minute.

### Folders
* docs: final project report in HTML format
* milestones: intermediate project reports in LaTex format
* intermediate files: pre-processed data

### Data Sources
* [Lending Club data set on Kaggle](https://www.kaggle.com/wendykan/lending-club-loan-data)
* [The Federal Reserve Bank of St. Louis - FRED database](https://fred.stlouisfed.org/)
