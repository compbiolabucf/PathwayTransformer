from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import get_dataset
from utils.flag import flag_bounded
from utils.lr import PolynomialDecayLR


def initialize_parameters(module: nn.Module, n_layers: int) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class PathwayTransformer(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        dataset_root,
        dirpath,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        dataset_bundle = get_dataset(dataset_name, dataset_root)
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.weight_decay = weight_decay
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.evaluator = dataset_bundle["evaluator"]
        self.metric = dataset_bundle["metric"]
        self.evaluator_metric = dataset_bundle.get("evaluator_metric", self.metric)
        self.loss_fn = dataset_bundle["loss_fn"]
        self.metrics_output_path = Path(dirpath) / "auroc.txt"

        self.atom_encoder = nn.Embedding(512 * 21 + 1, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(512 * 3 + 1, num_heads, padding_idx=0)
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                for _ in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, dataset_bundle["num_class"])
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.automatic_optimization = not self.flag
        self.validation_epoch_outputs = []
        self.test_epoch_outputs = []
        self.apply(lambda module: initialize_parameters(module, n_layers=n_layers))

    def encode_graph_batch(self, batch, perturb=None):
        attn_bias = batch.attn_bias
        spatial_pos = batch.spatial_pos
        node_features = batch.x
        in_degree = batch.in_degree
        out_degree = batch.out_degree
        edge_input = batch.edge_input
        attn_edge_type = batch.attn_edge_type

        n_graph, n_node = node_features.size()[:2]
        graph_attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).clone()

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] += spatial_pos_bias

        graph_token_distance = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] += graph_token_distance
        graph_attn_bias[:, :, 0, :] += graph_token_distance

        if self.edge_type == "multi_hop":
            clamped_spatial_pos = spatial_pos.clone()
            clamped_spatial_pos[clamped_spatial_pos == 0] = 1
            clamped_spatial_pos = torch.where(clamped_spatial_pos > 1, clamped_spatial_pos - 1, clamped_spatial_pos)
            if self.multi_hop_max_dist > 0:
                clamped_spatial_pos = clamped_spatial_pos.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_distance = edge_input.size(-2)
            flattened_edge_input = edge_input.permute(3, 0, 1, 2, 4).reshape(max_distance, -1, self.num_heads)
            flattened_edge_input = torch.bmm(
                flattened_edge_input,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_distance],
            )
            edge_input = flattened_edge_input.reshape(
                max_distance, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / clamped_spatial_pos.float().unsqueeze(-1)).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] += edge_input
        graph_attn_bias += attn_bias.unsqueeze(1)

        node_features = self.atom_encoder(node_features)
        node_features = node_features.sum(dim=-2)
        if self.flag and perturb is not None:
            node_features = node_features + perturb

        node_features = node_features + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).expand(n_graph, -1, -1)
        encoded_nodes = torch.cat([graph_token_feature, node_features], dim=1)

        hidden_states = self.input_dropout(encoded_nodes)
        for encoder_block in self.encoder_blocks:
            hidden_states = encoder_block(hidden_states, graph_attn_bias)
        return self.final_layer_norm(hidden_states)

    def forward(self, batch, perturb=None):
        encoded_graph = self.encode_graph_batch(batch, perturb)
        return self.output_projection(encoded_graph[:, 0, :])

    def training_step(self, batch, batch_idx):
        if not self.flag:
            predicted_logits = self(batch).view(-1)
            target_values = batch.y.view(-1).float()
            loss = self.loss_fn(predicted_logits, target_values)
        else:
            target_values = batch.y.view(-1).float()

            def model_forward(perturb):
                return self(batch, perturb)

            adversarial_forward = (self, model_forward)
            n_graph, n_node = batch.x.size()[:2]
            perturb_shape = (n_graph, n_node, self.hidden_dim)
            optimizer = self.optimizers()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = flag_bounded(
                adversarial_forward,
                perturb_shape,
                target_values,
                optimizer,
                batch.x.device,
                self.loss_fn,
                m=self.flag_m,
                step_size=self.flag_step_size,
                mag=self.flag_mag,
            )
            self.lr_schedulers().step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_logits = self(batch)
        target_values = batch.y
        validation_loss = self.loss_fn(predicted_logits.view(-1), target_values.view(-1).float())
        self.validation_epoch_outputs.append(
            {
                "y_pred": predicted_logits,
                "y_true": target_values,
                "val_loss": validation_loss.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if not self.validation_epoch_outputs:
            return
        predicted_logits = torch.cat([item["y_pred"] for item in self.validation_epoch_outputs])
        target_values = torch.cat([item["y_true"] for item in self.validation_epoch_outputs])
        validation_loss = torch.stack([item["val_loss"] for item in self.validation_epoch_outputs]).mean()
        self.validation_epoch_outputs.clear()

        self.log("val_loss", validation_loss, sync_dist=True)
        evaluation_input = {"y_true": target_values, "y_pred": predicted_logits}
        try:
            metric_value = self.evaluator.eval(evaluation_input)[self.evaluator_metric]
            self.log(f"valid_{self.metric}", metric_value, sync_dist=True)
        except Exception:
            pass

    def test_step(self, batch, batch_idx):
        self.test_epoch_outputs.append(
            {
                "y_pred": self(batch),
                "y_true": batch.y,
                "idx": batch.idx,
                "y_emb": self.encode_graph_batch(batch),
            }
        )

    def on_test_epoch_end(self):
        if not self.test_epoch_outputs:
            return

        predicted_logits = torch.cat([item["y_pred"] for item in self.test_epoch_outputs])
        target_values = torch.cat([item["y_true"] for item in self.test_epoch_outputs])
        encoded_embeddings = torch.cat([item["y_emb"] for item in self.test_epoch_outputs])
        sample_indices = torch.cat([item["idx"] for item in self.test_epoch_outputs])
        self.test_epoch_outputs.clear()

        _ = encoded_embeddings.cpu().float().numpy()
        _ = sample_indices

        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(
                f"model: {self.metrics_output_path.parent} AUROC: "
                f"{roc_auc_score(target_values.cpu(), predicted_logits.cpu())}\n"
            )

        evaluation_input = {"y_true": target_values, "y_pred": predicted_logits}
        self.log(f"test_{self.metric}", self.evaluator.eval(evaluation_input)[self.evaluator_metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PathwayTransformer")
        parser.add_argument("--n_layers", type=int, default=12)
        parser.add_argument("--num_heads", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--ffn_dim", type=int, default=512)
        parser.add_argument("--intput_dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--attention_dropout_rate", type=float, default=0.1)
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--warmup_updates", type=int, default=60000)
        parser.add_argument("--tot_updates", type=int, default=1000000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument("--edge_type", type=str, default="multi_hop")
        parser.add_argument("--validate", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument("--flag", action="store_true")
        parser.add_argument("--flag_m", type=int, default=3)
        parser.add_argument("--flag_step_size", type=float, default=1e-3)
        parser.add_argument("--flag_mag", type=float, default=1e-3)
        return parent_parser


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, ffn_size)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(ffn_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.output_projection(hidden_states)


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = attention_head_size = hidden_size // num_heads
        self.scale = attention_head_size**-0.5
        self.query_projection = nn.Linear(hidden_size, num_heads * attention_head_size)
        self.key_projection = nn.Linear(hidden_size, num_heads * attention_head_size)
        self.value_projection = nn.Linear(hidden_size, num_heads * attention_head_size)
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.output_projection = nn.Linear(num_heads * attention_head_size, hidden_size)

    def forward(self, query_states, key_states, value_states, attn_bias=None):
        original_query_shape = query_states.size()
        head_dim = self.attention_head_size
        batch_size = query_states.size(0)

        query_states = self.query_projection(query_states).view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key_states = self.key_projection(key_states).view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key_states = key_states.transpose(2, 3)
        value_states = self.value_projection(value_states).view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query_states * self.scale, key_states)
        if attn_bias is not None:
            attention_scores = attention_scores + attn_bias

        attention_probs = self.attention_dropout(torch.softmax(attention_scores, dim=3))
        context_states = attention_probs.matmul(value_states)
        context_states = context_states.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * head_dim)
        context_states = self.output_projection(context_states)
        assert context_states.size() == original_query_shape
        return context_states


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = GraphMultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size, ffn_size, dropout_rate)
        self.feed_forward_dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attn_bias=None):
        attention_input = self.self_attention_norm(hidden_states)
        attention_output = self.self_attention(attention_input, attention_input, attention_input, attn_bias)
        hidden_states = hidden_states + self.self_attention_dropout(attention_output)

        feed_forward_input = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(feed_forward_input)
        return hidden_states + self.feed_forward_dropout(feed_forward_output)
