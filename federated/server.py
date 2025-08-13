import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays
from model.resnet_model import get_resnet18_model

SAVE_DIR = "output"
SAVE_PATH = os.path.join(SAVE_DIR, "resnet18_global.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

def evaluate_aggregate(results):
    if not results:
        return {}
    agg = {}
    total = sum(n for n, _ in results)
    keys = set().union(*(m.keys() for _, m in results))
    for k in keys:
        s = sum(m.get(k, 0.0) * n for n, m in results)
        agg[k] = s / max(total, 1)
    if {"precision","f1_score","recall","accuracy"}.issubset(keys):
        print(f"[AGREGADO] recall={agg['recall']:.4f} | f1_score={agg['f1_score']:.4f} "
              f"| accuracy={agg['accuracy']:.4f} | precision={agg['precision']:.4f}")
    return agg

class SaveOnAggregate(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated
        params, _ = aggregated
        nds = parameters_to_ndarrays(params)
        model = get_resnet18_model(pretrained=False, num_classes=4)
        sd = model.state_dict()
        for (k, _), arr in zip(sd.items(), nds):
            if tuple(sd[k].shape) == tuple(arr.shape):
                sd[k] = torch.tensor(arr)
        model.load_state_dict(sd, strict=False)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Modelo global salvo em {SAVE_PATH} (round {server_round})")
        return aggregated

def get_strategy():
    return SaveOnAggregate(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=lambda r: evaluate_aggregate([(n, m) for n, m in r]),
    )

def main():
    print("Iniciando servidor Flower em localhost:8080")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=get_strategy(),
    )

if __name__ == "__main__":
    main()
