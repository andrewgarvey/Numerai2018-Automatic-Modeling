"""
Logistic Regression Numerai Model 

Andrew Garvey 
- Created Dec 2018
- Edited for dynamic local running Feb 2019
    - runs weekly - Sunday 3am (Currently monday for testing purposes) 
    - accounts for variable feature length and variable dependant variable lengths
    - makes its own folders to store results  
    - uploads csv automatically to my account,
    - email me upon completion
    - #Some way to long term store the results, maybe uploads a csv to ADD to it and the writes it back with a new name?
"""
# import packages
import numpy as np
import pandas as pd
import os
import datetime
import zipfile

#this is stupid 
import warnings 
warnings.filterwarnings("ignore") # yes i know this is kinda stupid 

# import modelling libraries
from sklearn import linear_model, model_selection


# Set up Connectio to my account for uploads 
import numerox as nx

NUMERAI_PUBLIC_ID = 'U2AEOL7YPSHHAQCW53J7GH2V33CXAXMR'
NUMERAI_SECRET_KEY ='ZG4GYBM5OCAR64JP6K3LXB35NQIO47JKKF5OJILCF3G6P2K5EUX3MPG5DK56IKKN'



#------------------------------------------------------------------------------
# directory and data
os.chdir('D:\\Numerai\\numerai_dataset')
data = nx.download("numerai_dataset.zip")   

#unzip file
zip_ref = zipfile.ZipFile('numerai_dataset.zip')
zip_ref.extract('numerai_training_data.csv',os.getcwd())
zip_ref.close()

# Import data 
train_data = pd.read_csv("numerai_training_data.csv", header=0)

# Drop elizabeth and jordan
train_data = train_data[train_data.columns.drop(list(train_data.filter(regex='elizabeth|jordan')))]

# Split out features 
X_train = train_data.filter(regex='feature') 
# split out targets
Y = train_data.filter(regex='target_')

#get tournaments , for backtest / production want only the names 
tournaments = list(Y.columns.str.replace('target_',''))



# Split out era for later grouping
era = train_data['era']
#------------------------------------------------------------------------------
# DIRECTORY management
# dynamic week + tournament name 
orig_date = datetime.date(2016,4,23)
new_date = datetime.date.today()
day_delta = (new_date-orig_date)
week_delta = round((day_delta.days)/7)

dmy = datetime.date.today().strftime("%d-%m-%Y")

# folder variables
DATE_TOURNAMENT = 'Tournament #' + str(week_delta) + ' ' + dmy  

# Make the new folders under submission using the date_tournament
os.chdir ('D:\\Numerai\\Submission')
try:  
    os.mkdir('D:\\Numerai\\Submission\\'+DATE_TOURNAMENT)
except OSError:
    print('already exists')

# set the directory to this week
os.chdir(os.path.join(os.getcwd(),DATE_TOURNAMENT))

#------------------------------------------------------------------------------
# Modeling Function
# Define the Logistic Regression Model,this is how numerai backtest / production needs the input

class logistic(nx.Model):

    def __init__(self, params):
        self.p = params

    def fit_predict(self, dfit, dpre, tournament):
        model = linear_model.LogisticRegression(C=self.p['C'], 
                                                solver=self.p['solver'], 
                                                multi_class=self.p['multi_class'],
                                                max_iter=self.p['max_iter'])
        model.fit(dfit.x, dfit.y[tournament])
        yhat = model.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat

#------------------------------------------------------------------------------
## Looping through tournaments
# All PARAMS 
        
# Define GridSearch Hyperparameters
C = [0.001,0.01,0.05]
solver = ["newton-cg", "lbfgs", "sag", "saga"]
multi_class = ["ovr", "multinomial"]
max_iter = [500]




hyperparameters = {'C': C,
                   'solver': solver,
                   'multi_class': multi_class,
                   'max_iter': max_iter}

#Define gridsearch split 
kfold_split = 2
groups=era

#Define gridsearch other
scoring = 'neg_log_loss'
random_state = 123
n_jobs = -1
verbose = 1

#LOOP
for index in range(0, len(tournaments)):
    # get the tournament name
    tournament = tournaments[index]
    
    print("*********** TOURNAMENT " + tournament + " ***********")
    
    # set the target name for the tournament
    target = "target_" + tournament 
    
    # set the y train with the target variable
    y_train = Y.iloc[:, Y.columns == target].values.reshape(-1,)
    
    # use GroupKFold for splitting the era
    group_kfold = model_selection.GroupKFold(n_splits=kfold_split).split(X_train,y_train,groups)

    print(">> finding best params")
    clf = model_selection.GridSearchCV(linear_model.LogisticRegression(random_state=random_state), 
                                       param_grid = hyperparameters, 
                                       scoring=scoring,
                                       cv=group_kfold,
                                       n_jobs=n_jobs,
                                       verbose=verbose)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print(">> best params: ", best_params)

    # create a new LR model for the tournament
    model = logistic(best_params)
    
    """
    print(">> training info:")
    train = nx.backtest(model, data, tournament, verbosity=1)
    """
    print(">> validation info:")
    validation = nx.production(model, data, tournament, verbosity=1)
    
    print(">> saving validation info: ")
    validation.to_csv(tournament + "-T" + str(week_delta)+ ".csv")
    print(">> done saving validation info")
    # Upload these results
    nx.upload(tournament + "-T" + str(week_delta)+ ".csv",tournament,NUMERAI_PUBLIC_ID,NUMERAI_SECRET_KEY)
    
    
    
#------------------------------------------------------------------------------
# Notify finished running + uploaded Results, whenever it is done

#setup
import smtplib

mail = smtplib.SMTP('smtp.gmail.com',587)
mail.ehlo()
mail.starttls()
mail.login('andrewtgarvey@gmail.com','Hszztqkz117')

Subject ='Numerai Results Uploaded'
Body = 'https://numer.ai/rounds' 
Message = 'Subject: {}\n\n{}'.format(Subject, Body)

mail.sendmail('andrewtgarvey@gmail.com','andrewtgarvey@gmail.com',Message)

mail.close()


#------------------------------------------------------------------------------
# Store Results
# Keep track of all the results somehow in a csv

#change directory to Summary
#os.chdir('D:\Numerai\Submission\Summary')
    
    
# ----------------------------------------------------------------------------
