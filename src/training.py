from sklearn.model_selection    import train_test_split
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import GridSearchCV
from pathlib                    import Path
from src.parser                 import YamlParser

import pandas as pd

YAML_FILE = Path(__file__).resolve().parent.parent / "config.yaml"
yaml_parser = YamlParser()
config = yaml_parser.load_yaml(YAML_FILE)

def create_variables(df: pd.DataFrame, feature_importances: bool, cols_fi: list[str] | None = None) -> train_test_split:
    """crea las variables para entrenar el modelo, con todas las columnas, o con las mas importantes

    Args:
        df (pd.DataFrame): data
        feature_importances (bool): False, no se hace nada. True, X seran las 15 variables con mas peso
        cols_fi (list[str] | None): columnas con mas peso. Puede llegar o no llegar-> None

    Returns:
        train_test_split: division de variables
    """
    if feature_importances:
        X = df[cols_fi]
    else:
        X = df.drop('Attrition', axis=1)

    y = df['Attrition']

    return train_test_split(
        X, y,
        test_size=      config['params']['test_size'],
        random_state=   config['params']['random_state'],
        stratify=y
    )

def training_model_rf(X_train, y_train) -> RandomForestClassifier:
    """Genera un modelo de RF optimizado y lo entrena

    Args:
        X_train (_type_): X_train
        y_train (_type_): y_train

    Returns:
        RandomForestClassifier: _description_
    """

    param_grid = {
        'n_estimators':     config['params']['n_estimators'],
        'max_depth':        config['params']['max_depth'],
        'min_samples_leaf': config['params']['min_samples_leaf']
    }

    grid_search = GridSearchCV(
        estimator=  RandomForestClassifier(random_state=config['params']['random_state'],class_weight=config['params']['class_weight_rfc']),
        param_grid= param_grid,
        scoring=    'recall',
        cv=         5,               
        n_jobs=     -1           # Usa todos los núcleos del procesador
    )

    grid_search.fit(X_train, y_train)

    #guardar el mejor modelo
    rf_optimizado = grid_search.best_estimator_

    return rf_optimizado
