import os
import pdb
import sys
from itertools import chain

import torch
import torch.nn as nn
from tac_grammar import CFG

from .tactic_decoder import TacticDecoder
from .term_encoder import TermEncoder

sys.path.append(os.path.abspath("."))
from time import time


class Prover(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.tactic_decoder = TacticDecoder(CFG(opts.tac_grammar, "tactic_expr"), opts)
        self.term_encoder = TermEncoder(opts)

    def create_batch(self, environment, local_context, goal):
        out = torch.tensor([], device=self.opts.device)

        idx = 0
        for env in chain(*environment):
            out = torch.hstack(
                (out, torch.ones(len(env["x"]) * idx, device=self.opts.device))
            )
            idx += 1

        for context in chain(*local_context):
            out = torch.hstack(
                (out, torch.ones(len(context["x"]) * idx, device=self.opts.device))
            )
            idx += 1

        out = torch.hstack(
            (out, torch.ones(len(goal["x"]) * idx, device=self.opts.device))
        )

        return out

    def embed_terms(self, batch):

        # if "gnn" in self.opts.encoder_model:  # Use GNN model to encode terms
        all_embeddings = self.term_encoder(batch)

        # reformat for decoder
        num_envs = len(batch.env['ast'])
        env_embed = torch.cat(  # environment
            [torch.zeros(3, device=self.opts.device), all_embeddings[0:num_envs]],
            dim=0
        )

        num_context = len(batch.local_context['ast'])
        context_embed = torch.cat(  # local context
            [torch.zeros(3, device=self.opts.device), all_embeddings[num_envs:num_envs+num_context]],
            dim=0
        )

        goal_embed = torch.cat(  # goal
            [torch.zeros(3, device=self.opts.device), all_embeddings[-1]],
            dim=0
        )

        # update padding references
        env_embed[:,0], context_embed[:,1], goal_embed[:,2] = 1.0, 1.0, 1.0

        return [env_embed], [context_embed], goal_embed

    # def forward(self, environment, local_context, goal, actions, teacher_forcing):
    def forward(self, batch, teacher_forcing):
        environment = batch["env"]
        local_context = batch["local_context"]
        goal = batch["goal"]
        actions = batch["actions"]
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
            "quantified_idents": [g.quantified_idents for g in goal],
        }
        asts, loss = self.tactic_decoder(
            environment, local_context, goal, actions, teacher_forcing
        )
        return asts, loss

    def beam_search(self, environment, local_context, goal):
        # TODO(danj/dhuang): update this to convert environment, local_context, goal
        # need to add the G_step to this method call
        d = {
            "env": [environment],
            "local_context": [local_context],
            "goal": [goal],
        }
        batch = None  # TODO: create this and combine with d
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
