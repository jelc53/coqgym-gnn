from torch import optim, nn
import lightning.pytorch as pl

class GNNTx(pl.LightningModule):
    def __init__(self, gnn, tx):
        super().__init__()
        self.gnn = gnn
        self.tx = tx

    def training_step(self, batch, batch_idx):
        x = self.gnn(batch.x, batch.edge_idx, batch.batch)
        x = self.tx(x)
        # loss calculation? LLM with teacher forcing, using x as context
        # self.log("train_loss", loss)
        # return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

class GNN(nn.Module):
    pass
