# %%
import sklearn
sklearn.__version__

# %%
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)

# %%
df
# %%

# predict survived -0 or 1
# classification problem because target is categorical
# columns except target are features, 10 obs

X = df[['Parch', 'Fare']]
X

# %%
X.shape
# %%
y=df['Survived']
y
# %%
y.shape
# %%
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'liblinear', random_state = 1)

#^ model object defined with solver and random state 
# for model eval need evaluation procedure [cross validation] and eval metric [classification =accuracy]
# using 3-fold cross validation - split rows in 3 subsets a b c
# a and b - training set, c - testing set, train on a and b, make prediction for c, evaluate c
# repeat process with other 2 subsets, each set becomes testing set once
# average the scores

from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean()
# can use classification metrics other than accuracy
# choosing model eval me


# %%
logreg.fit(X,y)
df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)

df_new
# %%
# ^ new dataset for which we don't know the target values

X_new = df_new[['Parch', 'Fare']]
X_new # data we make predictions on
# %%
logreg.predict(X_new)
# %%
# cross validation does do fitting but does not fit the final model to X and y
# part 2 , encoding of categorical data - to improve model
# add columns embarked and sex
# dummy encoding, one-hot encoding to encode categorical vars
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder()
ohe.fit_transform(df[['Embarked']])
# whats a sparse matrix- mostly 0- just store positions of non-zero and those vals
# what values in it simple encoded matrix made sparse
# meaning of fit_transform? fit is to learn C, S, Q values of embarked, tranform when it produces output matrix
# why 2 brackets? so that scikitlearn knows that dataframe with object is 2d object
# %%
ohe.fit_transform(df[['Embarked', 'Sex']])
# encoding these 2 to include in model with parched and fare
# %%
# pipeline so don't have to do task mutliple times
# different preprocessing on different columns via column transformer
# apply same workflow to training and testing sets via pipeline
cols =["Parch", "Fare", "Embarked", "Sex"]
X=df[cols]
X
# %%
# one hot encode embarked and sex and keep other 2 cols same
# column transformer will do this - apply diff preprocessing to diff cols
ohe = OneHotEncoder()
from sklearn.compose import make_column_transformer
ct = make_column_transformer((ohe, ['Embarked', 'Sex']), remainder='passthrough')
ct.fit_transform(X) #one hot encoding of 2 cols and stacks other 2 next
#col order, encoded ones then remaining ones
# %%
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(ct, logreg)
# ct for preprocessing, logreg for model building
# 2step pipe
pipe.fit(X, y)
# 4 cols of X passed to first steps, turned to 7 cols, they passed to logistic reg as feature matrix
# logreg is fit on transformed feauter matrix of 7 columsns along with y values
# %%
# same as
logreg.fit(ct.fit_transform(X), y)
# %%
pipe.named_steps.logisticregression.coef_
#these features relate to 7 feaures output by col tranformer
# %%
X_new = df_new[cols]
X_new
# %%
pipe.predict(X_new)
# %%
# what happen underneath
logreg.predict(ct.transform(X_new))
# %%
# transform - training scheme learned from training data- applied to new data
# first fit transform to learn the training scheme, then just fit to run on new data
# http://bit.ly/basic-pipeline
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

cols = ['Parch', 'Fare', 'Embarked', 'Sex']

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[cols]

ohe = OneHotEncoder()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    remainder='passthrough')

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
# %%
# when making a prediction, u get 1 prediction for every row
# so doing pipe.predict gets prediction per num rows
# hyper - parameter optimization and cross validation done of a pipeline, not put in one
# using custom functn - make custom transformer - stick it into a column transformer, put that col transformer in pipe
# encoding text data - next:
# H: name contains predictors for survival
# name is doctor or master. get on life boat first
df
# %%
from sklearn.feature_extraction.text import CountVectorizer
# convert words into matrix of token counts
vect = CountVectorizer()
#output document term matrix - sparse matrix, 10 rows, 40 cols, because countvectorizer created 40 features
dtm = vect.fit_transform(df['Name'])
dtm
# %%
print(vect.get_feature_names())
# %% gives bag of words representation
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
# %%
df.loc[0, 'Name']
# %%
# each word/column - column can learn relationshop between each row and predictor-
# tfidf is  CountVectorizer + some other tranfomations
# use transform not fit transform on test set
# can do text preprocessing in pipeline
# can be used instead of nltk to remove stopwords
# not NLP but ML only
cols = ["Parch", "Fare", "Embarked", "Sex", "Name"]
X = df[cols]
X
# %%
ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']), (vect, 'Name'),
    remainder='passthrough')

# %%
ct.fit_transform(X)
# %%
# 47 cols bcoz 3 for embarked 2 for sex 40 for name 1 for fare 1 for parched
pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)

# %%
pipe.named_steps
# %%
X_new= df_new[cols]
pipe.predict(X_new)
# %%
# need to vectorize 2 columns, make 2 vectorizers - one after the other in col trnasformer
# pipe structures possible 1. tranformer1, transfoermer2..., model 2. tranformer1, transformer2.....
# cross validation to check accuracy
# scaling can be done in column transformer or in midlde of pieline
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=10)
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest', nrows=10)
X_new = df_new[cols]

ohe = OneHotEncoder()
vect = CountVectorizer()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    remainder='passthrough')

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)
# %%
# NaN is missing value and different from data that might just be in test data but not in training data 
cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']
X = df[cols]
X

# %%
# will error out coz of NaN
pipe.fit(X, y)
# %%
X.dropna()
#will drop the row with Nan
# good option if very less training data dropped
# good option if missingness is completely random
# can mention subset param to drop na from just those cols
# can mention axis='columns' to drop column instead of  row

# impute missing vals
# fill with known data


# %%
# imputing missing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
imp.fit_transform(X[['Age']])
# fills in mean in the missing vals
# also support median, mode or user defined val
# %%
imp.statistics_
# %%
#simple imputer can be used for categoricals
ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    remainder='passthrough')
ct.fit_transform(X)
#48 instead of 47 cols returned bcoz age is now part
# %%
pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.named_steps
# %%
X_new = df_new[cols]
pipe.predict(X_new)
# %%
#if X-new doesnt have NAs then nothing gets imputed in test data
# If it does, then replacement values are means from X not X-new
# bcoz learning things only from training data and then apply learned things to trainng+test data
# thats why we fit_transform training data , and only transform test dataset
# pipe.predict makes predications one row at a time [only makes sens to get value from training data coz dont have other values of test data]

# when imputing missing vals, can add missingness as a feature
# by adding binray matrix indicating presence of missing values
imp_indicator = SimpleImputer(add_indicator=True)
imp_indicator.fit_transform(X[['Age']])
# preserves indicators/info abt which were missing vals- that info might have relation with target
# ie missing -yes/no itself becomse predictor of the DV
# missingness becaomes a feature passed to model, useful esp when data not randomly missing
# MNAR - missing not at random
# %%
# KNN imputer or iterative imputer might be better in some cases
# incl multple models in a pipline - using stacking classifier or voting classifier
# put that classifier as last in pipleine
# but cant compare outputs of those 2 model

#whether feature is relevant to put in model - use stats tests, or automated feature selection, etc
# ohe creates new column, same thing with imputer 

# swutch to full dataset

df = pd.read_csv('http://bit.ly/kaggletrain')
df.shape
# %%
df_new = pd.read_csv('http://bit.ly/kaggletest')
df_new.shape
# 11 cols in test and 12 cols in train because test dodesnt have target val

# %%


df.isna().sum()
# %%
df_new.isna().sum()
# missing values in ttest, not train data
# missing vals in train, not test dataset

X= df[cols]
y = df['Survived']
ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    remainder='passthrough')

# ohe will throw error because of missing vals in embarke
# cant impute mean or mediam of embarked coz it is str
# can impute mode
# or a contt value of choice 

imp_constant = SimpleImputer(strategy = 'constant', fill_value='missing')
# treats missing val as 4th category, so 4 columns after ohe
# similar to imputer + indicatir

# transformeer only pipelne
imp_ohe = make_pipeline(imp_constant, ohe)
imp_ohe.fit_transform(X[['Embarked']])
# %%
# pipe ends in model, use fit
# pipe ends in transformer, use fit_transform
ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    remainder='passthrough')
# imputer affects only embarked, not sex, coz only embarked has missing vals
# imp_ohe is a pipleine full of 2 trnformees and acts as a tranformer

ct.fit_transform(X)

# %%

# now fix fare haiving missing vals in test not train set
ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    remainder='passthrough')

# calc imp value and test data takes that for NAs

ct.fit_transform(X)
# %%
pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
X_new = df_new[cols]
pipe.predict(X_new)

# %%
# imputed vals:
ct.named_transformers_.simpleimputer.statistics_
# %%
ct.named_transformers_.pipeline.named_steps.simpleimputer.statistics_
# %%
# recap

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

df = pd.read_csv('http://bit.ly/kaggletrain')
X = df[cols]
y = df['Survived']

df_new = pd.read_csv('http://bit.ly/kaggletest')
X_new = df_new[cols]

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder()

imp_ohe = make_pipeline(imp_constant, ohe)
vect = CountVectorizer()
imp = SimpleImputer()

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    remainder='passthrough')

logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)
pipe.predict(X_new)


# %%
# cant have imp and ohe separately 
# if same columns need to have 2+ operations, just put the operations in a pipeline and pipelien in ct
# drawbacks for pandas - cannot do count vectorizer, missing value imputaion using pandas- will have data leakage

# what is data leakage 
# model valuations procesudre to check whther overfitting training dataset
# imputation values in pandas is calculated on baiss of all data. Learning from testing data- that you not allowed to know [data leakage]
# data leage bad- bcoz get to know something not allowed to kneo

# cross validation unrelaible with 10 rows
# coss validation part:
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv = 5, scoring = 'accuracy').mean()
# ^ cross_val_score avoids data leakage by doing the split, then transformation, so only learning from training set
# use cross_val_score on pipeline to avoid data leakage

# tuning topic:
# we are using default hyperparameters for everyting in the pipeline
# we might get better scores by changing hyperparams for models and transformers
# tune hyperparams of logistic reg and countvectorizer
# use scikitlearn and can tune imputaion strategy at the same time as tuning the model
# using gridsearch - define vals wanna try for hyperparams, it cross-validates all possible combos of those vals
# hyperparams are vals u set, params are vals learned by model during fitting

#%%
pipe.named_steps.keys()
# %%
# specifying params that you want to tune

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params
# %%
# how do i know what to tune - needs experiene and research
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring = 'accuracy')
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results
# %%
# take values corresponding to highest mean-test-score
results.sort_values('rank_test_score')

# tune transformers
pipe.named_steps.columntransformer.named_transformers_
# %%
# tune dropeed param of OneHotEncoder
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]
params

# %%
grid = GridSearchCV(pipe, params, cv=5, scoring = 'accuracy')
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results


# %%
results.sort_values('rank_test_score')
# %%
grid.best_score_
grid.best_params_
# %%
grid.predict(X_new)
# grid automatically refits X and y using best params
# %%
