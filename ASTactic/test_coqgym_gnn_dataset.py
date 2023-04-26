from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from coqgym_gnn_dataset import CoqGymGNNDataset
from pathlib import Path


def merge(proof_steps):
    fields = [
        "file",
        "proof_name",
        "n_step",
        "env",
        "local_context",
        "goal",
        "is_synthetic",
        "tactic_actions",
        "tactic_str",
    ]
    meta_data = {key: [] for key in fields}
    for proof_step in proof_steps:
        for key, value in proof_step._store.items():
            if key not in fields:
                continue
            meta_data[key].append(value)
    batch = Batch.from_data_list(proof_steps)
    for k, v in meta_data.items():
        batch[k] = v
    return batch


def test_dataset():
    paths = list(Path("proof_steps/train").glob("*.pt"))
    paths += list(Path("proof_steps/valid").glob("*.pt"))
    # NOTE: can filter here...
    ds = CoqGymGNNDataset(paths)
    batch_size = 4
    loader = DataLoader(
        ds,
        shuffle=True,
        num_workers=1,
        collate_fn=merge,
        batch_size=batch_size,
    )
    batch = next(iter(loader))
    assert len(batch) == batch_size, "incorrect length!"
