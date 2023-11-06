from sklearn.model_selection import train_test_split , cross_validate , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from util import update_model ,save_simple_metrics_report ,get_model_performance_test_set


import logging
import sys 
import numpy as np 
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger=logging.getLogger(__name__)

logger.info('Loading Data...')

data= pd.read_csv('dataset/fulldata.csv')

logger.info('Loading Model...')

model = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('core_model', GradientBoostingRegressor())
])

logger.info('Separating Dataset into train and test')

X = data.drop(['worldwide_gross'], axis = 1)
y = data['worldwide_gross']

x_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.35,random_state=83)

logger.info('Setting Hyperparameters to Tune')

param_tunning = {'core_model__n_estimators': range(20,501,20)} 

gs=GridSearchCV(model, param_grid=param_tunning,scoring='r2',cv=5)

logger.info('Starting Grid Search....')

gs.fit(x_train,y_train)

logger.info('Cross Validating with Best Model...')

final_result=cross_validate(gs.best_estimator_,x_train,y_train,return_train_score=True,cv=5)

train_score=np.mean(final_result['train_score'])
validation_score=np.mean(final_result['test_score'])

assert train_score > 0.7
assert validation_score > 0.65 

logger.info(f'Train Score : {train_score:.3f}')
logger.info(f'Validation Score : {validation_score:.3f}')

logger.info('Updating model...')
update_model(gs.best_estimator_)

test_score = gs.best_estimator_.score(x_test,y_test)

logger.info('Generating a Model Reports..')
save_simple_metrics_report(train_score,test_score,validation_score,gs.best_estimator_)

y_test_pred = gs.best_estimator_.predict(x_test)
get_model_performance_test_set(y_test,y_test_pred)

logger.info('Training Finished')







