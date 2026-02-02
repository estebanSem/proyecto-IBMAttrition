from sklearn.base       import BaseEstimator, TransformerMixin
import numpy            as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Crea varaibles extras en el DataFrame

        Args:
            X (_type_): Data

        Returns:
            X_Copy: Dataframe con las nuevas variables
        """

        X_copy = X.copy()

        # Ratio de salario por año trabajado
        X_copy['Income_per_Year']       =   X_copy['MonthlyIncome'] / (X_copy['TotalWorkingYears'] + 1)

        # Años en la empresa vs Edad
        X_copy['Tenure_Age_Ratio']      =   X_copy['YearsAtCompany'] / (X_copy['Age'] + 1)

        # Ratio de compañias en las que ha trabajado.
        X_copy['ratio_years_company']   =   X_copy['NumCompaniesWorked'] / X_copy['TotalWorkingYears']
        X_copy['ratio_years_company']   =   X_copy['ratio_years_company'].replace([np.inf, -np.inf], 0)

        return X_copy