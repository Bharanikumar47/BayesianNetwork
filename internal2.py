import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
names=["age","chol","target","fbs"]
data1={
    "age":[17,34,37,45,57,64,35,67,42,45,67,78,23,88,97,53,67,32],
    "chol":[0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,0,0],
    "target":[0,0,0,1,1,1,0,1,0,0,1,1,0,0,1,1,0,1],
    "fbs":[0,0,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,1],
    }
data=pd.DataFrame(data1)
data.to_csv("Heart_disease.csv",index=False)
heart_disease=pd.read_csv("Heart_disease.csv")
model=BayesianNetwork([("age","target"),
       ("target","chol"),("chol","fbs")])
model.fit(heart_disease,estimator=MaximumLikelihoodEstimator)
var_est=VariableElimination(model)
q=var_est.query(variables=["target"],evidence={"age":37})
print(q)

