import os
import shutil

ORIGEM_TRAIN = "dataset_kaggle/Training"
ORIGEM_TEST = "dataset_kaggle/Testing"
DESTINO = "client2_original"

def copiar_imagens(origem, destino):
    for classe in os.listdir(origem):
        pasta_origem = os.path.join(origem, classe)
        pasta_destino = os.path.join(destino, classe)
        os.makedirs(pasta_destino, exist_ok=True)

        for arquivo in os.listdir(pasta_origem):
            origem_arquivo = os.path.join(pasta_origem, arquivo)
            destino_arquivo = os.path.join(pasta_destino, arquivo)
            shutil.copy2(origem_arquivo, destino_arquivo)

if __name__ == "__main__":
    os.makedirs(DESTINO, exist_ok=True)
    copiar_imagens(ORIGEM_TRAIN, DESTINO)
    copiar_imagens(ORIGEM_TEST, DESTINO)
    print("Imagens copiadas para", DESTINO)
