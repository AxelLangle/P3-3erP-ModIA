import pandas as pd


def construir_cubo_olap(resultados):
    """
    Construye un cubo OLAP a partir de resultados de evaluación.

    Args:
        resultados: lista de dicts con keys:
            - algoritmo: str ('KNN', 'K-Means', 'DBSCAN')
            - tipo_prueba: str ('Robustez', 'Confusión')
            - precision, recall, f1_score, accuracy: float (porcentajes)

    Returns:
        DataFrame con dimensiones: algoritmo, tipo_prueba, metrica, valor
    """
    df = pd.DataFrame(resultados)

    id_vars = ['algoritmo', 'tipo_prueba']
    value_vars = [c for c in df.columns if c not in id_vars]

    cube = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='metrica',
        value_name='valor'
    )

    return cube


def consultar_cubo(cube, algoritmo=None, tipo_prueba=None, metrica=None):
    """Filtra el cubo OLAP por una o más dimensiones."""
    result = cube.copy()
    if algoritmo:
        result = result[result['algoritmo'] == algoritmo]
    if tipo_prueba:
        result = result[result['tipo_prueba'] == tipo_prueba]
    if metrica:
        result = result[result['metrica'] == metrica]
    return result


def pivot_cubo(cube, filas='algoritmo', columnas='metrica',
               valores='valor', filtro_tipo=None):
    """Genera una tabla pivote del cubo OLAP para análisis comparativo."""
    data = cube.copy()
    if filtro_tipo:
        data = data[data['tipo_prueba'] == filtro_tipo]
    if data.empty:
        return pd.DataFrame()
    return data.pivot_table(
        index=filas, columns=columnas,
        values=valores, aggfunc='mean'
    )
