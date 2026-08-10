import argparse

print("El script comenzó a ajecutarse...")

def main():
    print("Main Iniciado")
    parser = argparse.ArgumentParser(description="Pruebas de detección facial")
    parser.add_argument('--fase', choices=['embeddings', 'clustering', 'classification', 'visualization', 'dashboard'])
    args = parser.parse_args()

    if args.fase == 'embeddings':
        from embeddings.core import generar_embeddings
        generar_embeddings()

    elif args.fase == 'clustering':
        from clustering.dbscan_cluster import agrupar_rostros
        agrupar_rostros()

    elif args.fase == 'visualization':
        from visualization.umap_3d import visualizar_espacio_latente
        visualizar_espacio_latente()

    elif args.fase == 'classification':
        from classification.knn_classifier import entrenar_clasificador
        entrenar_clasificador()

    elif args.fase == 'dashboard':
        from dashboard.visor_identidades_app import obtener_embedding
        obtener_embedding()

if __name__ == "__main__":
        print("Ejecutando el script principal...")
        main()