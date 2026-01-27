import pandas     as pd
import numpy      as np

from sklearn.feature_selection  import SelectFromModel
from sklearn.model_selection    import GridSearchCV, train_test_split
from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble           import RandomForestClassifier
from src.processing             import drop_columns,drop_extra_columns,get_more_two_values_cols,get_binary_cols,get_one_value_col
from src.data_loader            import create_df
from src.feature_engeneering    import FeatureEngineer
from sklearn.linear_model       import LogisticRegression
from pathlib                    import Path
from src.parser                 import YamlParser
from sklearn.preprocessing      import StandardScaler
from imblearn.combine           import SMOTEENN
from imblearn.pipeline          import Pipeline as ImPipeline

YAML_FILE = Path(__file__).resolve().parent.parent / "config.yaml"
yaml_parser = YamlParser()
config = yaml_parser.load_yaml(YAML_FILE)


df = create_df('raw_data')
df = df.drop(columns=['Attrition'])

one_value_cols    = get_one_value_col(df)
binary_cols       = get_binary_cols(df)
dummies           = get_more_two_values_cols(df)
extra_cols        = ['DailyRate','MonthlyRate','HourlyRate','EmployeeNumber','JobLevel','YearsInCurrentRole','TotalWorkingYears']
log_cols          = ['MonthlyIncome']

def create_variables(df: pd.DataFrame) -> train_test_split:
    """crea las variables para entrenar el modelo, con todas las columnas, o con las mas importantes
    Args:
        df (pd.DataFrame): data
        feature_importances (bool): False, no se hace nada. True, X seran las 15 variables con mas peso
        cols_fi (list[str] | None): columnas con mas peso. Puede llegar o no llegar-> None
    Returns:
        train_test_split: division de variables
    """
    X = df.drop('Attrition', axis=1)
    y = df['Attrition'].map({'Yes': 1, 'No': 0})

    return train_test_split(
        X, y,
        test_size       = config['params']['test_size'],
        random_state    = config['params']['random_state'],
        stratify        = y
    )

def data_preprocessing() -> ColumnTransformer :
    """ColumnTransformer para el preprocesado de los datos

    Returns:
        ColumnTransformer: Transformador de columnas
    """

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

def final_pipeline(f_importance:    bool = False, 
                   random_forest:   bool = False, 
                   logistic_reg:    bool = False,
                   smote:           bool = False,
                   coef:            bool = False) -> Pipeline :
    """_summary_

    Args:
        f_importance (bool, optional): _description_. Defaults to False.
        random_forest (bool, optional): _description_. Defaults to False.
        logistic_reg (bool, optional): _description_. Defaults to False.
        smote (bool, optional): _description_. Defaults to False.

    Returns:
        Pipeline: _description_
    """
    steps = [
        ('engineering', FeatureEngineer()), #creo las nuevas variables
        ('preprocessing', data_preprocessing()), #se hace el preprocesado
    ]

    #añado a steps uno de los modelos.
    #IMPORTANTE: En cualquiera de los casos, el ultimo APPEND ha de ser el del modelo
    if random_forest:

        if f_importance:
            #se hace un preentreno del modelo, para saber las 15 variables con mas peso
            steps.append(
                ('feature_importances', SelectFromModel(
                            RandomForestClassifier(
                                random_state=42,
                                class_weight='balanced'
                            ),
                        max_features=15,
                        threshold=-np.inf
                    )
                )
            )

        steps.append(
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced')) #entreno del modelo
        )

        model_pipeline = Pipeline(
            steps=steps
        )

    if logistic_reg:

        if smote:
            steps.append(('smote', SMOTEENN(random_state=42)))

        steps.append(('scaler', StandardScaler()))

        if coef:
            steps.append(
                ('coef', SelectFromModel(
                        LogisticRegression(
                            penalty     ='l1', 
                            solver      ='liblinear',
                            random_state= 42,
                            class_weight='balanced'
                        ),
                        max_features    = 15,
                        threshold       = -np.inf
                    )
                )
            )

        steps.append(
            ('model', LogisticRegression(
                    random_state    =   config['params']['random_state'],
                    class_weight    =   'balanced'
                )
            ) #entreno del modelo
        )

        model_pipeline = ImPipeline(
            steps=steps
        )


    return model_pipeline


def apply_grid_search(pipeline: Pipeline) -> RandomForestClassifier:
    """Genera un modelo de RF optimizado y lo entrena

    Args:
        X_train (_type_): X_train
        y_train (_type_): y_train

    Returns:
        RandomForestClassifier: _description_
    """

    param_grid = {
        'model__n_estimators'      : config['params']['n_estimators'],
        'model__max_depth'         : config['params']['max_depth'],
        'model__min_samples_leaf'  : config['params']['min_samples_leaf']
    }

    grid_search = GridSearchCV(
        estimator   = pipeline,
        param_grid  = param_grid,
        scoring     = config['params']['recall'],
        cv          = 5,               
        n_jobs      = -1           # usa todos los núcleos del procesador
    )

    return grid_search