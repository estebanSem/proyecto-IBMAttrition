import pandas     as pd
import numpy      as np

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble         import RandomForestClassifier
from src.processing           import apply_log_transform,drop_columns,drop_extra_columns,get_more_two_values_cols,get_binary_cols,get_one_value_col
from src.data_loader          import create_df


df = create_df('raw_data')
df = df.drop(columns=['Attrition'])

one_value_cols    = get_one_value_col(df)
binary_cols       = get_binary_cols(df)
dummies           = get_more_two_values_cols(df)
extra_cols        = ['DailyRate','MonthlyRate','HourlyRate','EmployeeNumber','JobLevel','YearsInCurrentRole','TotalWorkingYears']
log_cols          = ['MonthlyIncome']

def data_preprocessing() -> ColumnTransformer : 
  preprocesador = ColumnTransformer(
      transformers = [
          #elimina las columnas 
          ('drop_columns','drop',one_value_cols),

          #elimina otras columnas irrelevante
          ('drop_extra_columns', 'drop', extra_cols),

          #transformacion logaritmica del sueldo
          ('log_transform', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), log_cols),

          #binary / label encoding
          ('biniary_enc', OrdinalEncoder(), binary_cols),

          #dummies
          ('oneHot', OneHotEncoder(drop='first', sparse_output=False), dummies)
      ],

      remainder='passthrough' # El resto de columnas (numéricas) se quedan igual
  )

  return preprocesador

def final_pipeline() -> Pipeline :

    model_pipeline = Pipeline(
      steps=[
        ('engineering', FeatureEngineer()),
        ('preprocessing', data_preprocessing()),
        ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
      ]
    )

    return model_pipeline


