#funciones de entrenamiento para los dos modelos, RandomForest y LogisticRegression
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

#obtenemos el csv en crudo
df = create_df('raw_data')
df = df.drop(columns=['Attrition'])


def setup_data() -> tuple:

    #columnas con un valor
    one_value_cols    = get_one_value_col(df)

    #columnas binarias
    binary_cols       = get_binary_cols(df)

    #columnas a transformar con oneHot
    dummies           = get_more_two_values_cols(df)

    #columnas con correlacion alta y sin valor predictivo
    extra_cols        = ['DailyRate','MonthlyRate','HourlyRate','EmployeeNumber','JobLevel','YearsInCurrentRole','TotalWorkingYears']

    #columnas para escala logaritmica
    log_cols          = ['MonthlyIncome']

    return one_value_cols, binary_cols, dummies, extra_cols, log_cols

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
        test_size       = config['params']['trainning']['test_size'],
        random_state    = config['params']['trainning']['random_state'],
        stratify        = y
    )

def data_preprocessing() -> ColumnTransformer :
    """ColumnTransformer para el preprocesado de los datos

    Returns:
        ColumnTransformer: Transformador de columnas
    """
    #obtener columnas a tratar
    one_value_cols, binary_cols, dummies, extra_cols, log_cols = setup_data()


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

def final_pipeline(model_type:          str,
                   use_importances:     bool = False,
                   use_smote:           bool = False) -> ImPipeline :
    """Crea pipelines segun el tipo de modelo y segun las necesidades especificadas.
        SMOTE solo se aplicara a Regresion Logistica
        Se podran entrenar los modelos con todas o las 15 variables con mas peso

    Args:
        f_importance (bool, optional): _description_. Defaults to False.
        random_forest (bool, optional): _description_. Defaults to False.
        logistic_reg (bool, optional): _description_. Defaults to False.
        smote (bool, optional): _description_. Defaults to False.

    Returns:
        Pipeline: _description_
    """
    #1. primeros dos steps
    steps = [
        ('engineering', FeatureEngineer()), #creo las nuevas variables
        ('preprocessing', data_preprocessing()), #se hace el preprocesado
    ]

    #IMPORTANTE: En cualquiera de los casos, el ultimo APPEND ha de ser el del modelo
    if model_type == config['params']['model_type']['random_forest']:

        #Predefino class_weight para que al usar SMOTE, no aplicarle 'balanced'
        class_weight = config['params']['trainning']['class_weight_rfc']

        if use_smote:
            steps.append(('smote', SMOTEENN(random_state=config['params']['trainning']['random_state'])))
            #asignamos None porque usamos SMOTE 
            class_weight = None

        #3.entreno del modelo
        steps.append(
            ('model', RandomForestClassifier(random_state=config['params']['trainning']['random_state'], class_weight=class_weight)) #entreno del modelo
        )

        model_pipeline = ImPipeline(
            steps=steps
        )

    elif model_type == config['params']['model_type']['logistic_regression']:

        #2.añadimos SMOTE al pipeline si se requiere
        if use_smote:
            steps.append(('smote', SMOTEENN(random_state=config['params']['trainning']['random_state'])))

        #3.escalado de datos
        steps.append(('scaler', StandardScaler()))

        #4.se hace un preentreno del modelo, para saber las 15 variables con mas peso
        if use_importances:
            steps.append(
                ('coef', SelectFromModel(
                        LogisticRegression(
                            penalty     ='l1', 
                            solver      ='liblinear',
                            random_state= config['params']['trainning']['random_state'],
                            class_weight= config['params']['trainning']['class_weight_lr']
                        ),
                        max_features    = 15,
                        threshold       = -np.inf
                    )
                )
            )

        #5.entreno del modelo
        steps.append(
            ('model', LogisticRegression(
                    random_state    =   config['params']['trainning']['random_state'],
                    class_weight    =   config['params']['trainning']['class_weight_lr']
                )
            )
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
        'model__n_estimators'      : config['params']['trainning']['n_estimators'],
        'model__max_depth'         : config['params']['trainning']['max_depth'],
        'model__min_samples_leaf'  : config['params']['trainning']['min_samples_leaf']
    }

    grid_search = GridSearchCV(
        estimator   = pipeline,
        param_grid  = param_grid,
        scoring     = config['params']['trainning']['recall'],
        cv          = 5,               
        n_jobs      = -1           # usa todos los núcleos del procesador
    )

    return grid_search