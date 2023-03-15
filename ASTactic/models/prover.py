import os
import pdb
import sys
from itertools import chain

import torch
import torch.nn as nn
from tac_grammar import CFG

from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

import networkx as nx

from .tactic_decoder import TacticDecoder
from .term_encoder_stackgnn import TermEncoder

sys.path.append(os.path.abspath("."))
from time import time


def to_nx_graph(term):
    G = nx.Graph()
    for i, x in enumerate(term["x"]):
        G.add_node(i, x=x)
    G.add_edges_from(term["edge_index"].T.numpy())
    return G


class Prover(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.tactic_decoder = TacticDecoder(CFG(opts.tac_grammar, "tactic_expr"), opts)
        self.term_encoder = TermEncoder(opts)

    def embed_terms(self, batch):
        environment_embeddings = []
        context_embeddings = []
        goal_embeddings = []

        for proof_step in batch.to_data_list():

            # generate embeddings
            embeddings = self.term_encoder(proof_step)

            # environment
            n_env = len(proof_step.env)
            environment_embeddings.append(
                torch.cat(
                    [
                        torch.zeros((n_env, 3), device=self.opts.device),
                        embeddings[0:n_env]
                    ],
                    dim=1
                )
            )
            environment_embeddings[-1][:, 0] = 1.0

            # local context
            n_cxt = len(proof_step.local_context)
            context_embeddings.append(
                torch.cat(
                    [
                        torch.zeros((n_cxt, 3), device=self.opts.device),
                        embeddings[n_env: n_env + n_cxt],
                    ],
                    dim=1,
                )
            )
            context_embeddings[-1][:, 1] = 1.0

            # goal
            goal_embeddings.append(
                torch.cat(
                    [
                        torch.zeros(3, device=self.opts.device),
                        embeddings[-1]
                    ],
                    dim=0
                )
            )
            goal_embeddings[-1][2] = 1.0

        goal_embeddings = torch.stack(goal_embeddings)

        return environment_embeddings, context_embeddings, goal_embeddings

    # def forward(self, environment, local_context, goal, actions, teacher_forcing):
    def forward(self, batch, teacher_forcing):
        environment = batch["env"]
        local_context = batch["local_context"]
        goal = batch["goal"]
        actions = batch["tactic_actions"]
        environment_embeddings, context_embeddings, goal_embeddings = self.embed_terms(
            batch
        )
        environment = [
            {
                "idents": [v["qualid"] for v in env],
                "embeddings": environment_embeddings[i],
                "quantified_idents": [v["ast"].quantified_idents for v in env],
            }
            for i, env in enumerate(environment)
        ]
        local_context = [
            {
                "idents": [v["ident"] for v in context],
                "embeddings": context_embeddings[i],
                "quantified_idents": [v["ast"].quantified_idents for v in context],
            }
            for i, context in enumerate(local_context)
        ]
        goal = {
            "embeddings": goal_embeddings,
            "quantified_idents": [g['ast'].quantified_idents for g in goal],
        }
        asts, loss = self.tactic_decoder(
            environment, local_context, goal, actions, teacher_forcing
        )
        return asts, loss

    def beam_search(self, environment, local_context, goal):
        # need to add the G_step to this method call
        proof_step = {
            "env": environment,
            "local_context": local_context,
            "goal": goal,
        }
        Gs = []
        for env in proof_step["env"]:
            G = to_nx_graph(env)
            Gs.append(G)
        for lc in proof_step["local_context"]:
            G = to_nx_graph(lc)
            Gs.append(G)
        Gs.append(to_nx_graph(proof_step["goal"]))
        Gs = [from_networkx(G) for G in Gs]
        B = Batch.from_data_list(Gs)
        for k, v in proof_step.items():
            if k not in ['x', 'edge_index']:
                B[k] = v
        batch = Batch.from_data_list([B])
        environment_embeddings, context_embeddings, goal_embeddings = self.embed_terms(
            batch
        )
        environment = {
            "idents": [v["qualid"] for v in environment],
            "embeddings": environment_embeddings[0],
            "quantified_idents": [v["ast"].quantified_idents for v in environment],
        }
        local_context = {
            "idents": [v["ident"] for v in local_context],
            "embeddings": context_embeddings[0],
            "quantified_idents": [v["ast"].quantified_idents for v in local_context],
        }
        goal = {
            "embeddings": goal_embeddings,
            "quantified_idents": goal["ast"].quantified_idents,
        }
        asts = self.tactic_decoder.beam_search(environment, local_context, goal)
        return asts
