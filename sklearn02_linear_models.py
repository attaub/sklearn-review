############################################
import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(,)
reg.coef_
reg.intercept_
reg.predict()


reg = linear_model.BayesianRidge()
reg.fit()
reg.predict()
reg.coef_


from sklearn.linear_model import TweedieRegressor
reg = TweedieRegressor(power=1, alpha=0.5, link='log')
reg.fit()
reg.coef_
reg.intercept_
