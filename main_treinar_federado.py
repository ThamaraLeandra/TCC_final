from federated.client import start_client
from federated.server import start_server
import multiprocessing
import time

def start_client1():
    start_client("client1_preprocessed")

def start_client2():
    start_client("client2_original")

def main():
    server_process = multiprocessing.Process(target=start_server)
    client1_process = multiprocessing.Process(target=start_client1)
    client2_process = multiprocessing.Process(target=start_client2)

    server_process.start()
    time.sleep(2)  # Espera para garantir que o servidor suba antes dos clientes
    client1_process.start()
    client2_process.start()

    server_process.join()
    client1_process.join()
    client2_process.join()

if __name__ == "__main__":
    main()
