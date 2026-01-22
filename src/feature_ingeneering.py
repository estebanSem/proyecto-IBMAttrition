from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self # No hace nada en el fit
        
    def transform(self, X):
        X_copy = X.copy()
        # Ejemplo: Ratio de salario por año trabajado
        X_copy['Income_per_Year'] = X_copy['MonthlyIncome'] / (X_copy['TotalWorkingYears'] + 1)
        # Ejemplo: Años en la empresa vs Edad
        X_copy['Tenure_Age_Ratio'] = X_copy['YearsAtCompany'] / (X_copy['Age'] + 1)
        X_copy['ratio_years_company'] = X_copy['NumCompaniesWorked'] / X_copy['TotalWorkingYears']
        X_copy['ratio_years_company'] = X_copy['ratio_years_company'].replace([np.inf, -np.inf], 0)
        return X_copy