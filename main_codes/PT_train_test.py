import logging
import os
import warnings
from argparse import ArgumentParser

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from data import GraphDataModule, get_dataset
from model import PathwayTransformer


def create_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_root_dir", type=str, default="exps/default")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--enable_checkpointing", action="store_true", default=True)
    early_stop_group = parser.add_mutually_exclusive_group()
    early_stop_group.add_argument("--enable_early_stopping", dest="enable_early_stopping", action="store_true")
    early_stop_group.add_argument("--disable_early_stopping", dest="enable_early_stopping", action="store_false")
    parser.set_defaults(enable_early_stopping=True)
    PathwayTransformer.add_model_specific_args(parser)
    GraphDataModule.add_argparse_args(parser)
    return parser


def create_trainer_kwargs(args, callbacks):
    devices = args.devices if args.gpus is None else args.gpus
    return {
        "accelerator": args.accelerator,
        "devices": devices,
        "strategy": args.strategy,
        "precision": args.precision,
        "default_root_dir": args.default_root_dir,
        "max_epochs": args.max_epochs,
        "max_steps": args.max_steps,
        "log_every_n_steps": args.log_every_n_steps,
        "deterministic": args.deterministic,
        "callbacks": callbacks,
        "enable_checkpointing": args.enable_checkpointing,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }


def configure_runtime() -> None:
    warnings.filterwarnings("ignore", message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*The `srun` command is available on your system but is not used.*")
    warnings.filterwarnings("ignore", message=r".*Precision 16-mixed is not supported by the model summary.*")
    warnings.filterwarnings("ignore", message=r".*Tensor Cores.*torch.set_float32_matmul_precision.*")
    warnings.filterwarnings("ignore", message=r".*The given NumPy array is not writable.*")
    warnings.filterwarnings("ignore", message=r".*Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.*")
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class EpochMetricsPrinter(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss is None or val_loss is None:
            return
        epoch_index = trainer.current_epoch
        print(f"epoch {epoch_index} train_loss: {float(train_loss):.6f} val_loss: {float(val_loss):.6f}")


def create_model(arguments, data_module):
    model_init_args = dict(
        n_layers=arguments.n_layers,
        num_heads=arguments.num_heads,
        hidden_dim=arguments.hidden_dim,
        attention_dropout_rate=arguments.attention_dropout_rate,
        dropout_rate=arguments.dropout_rate,
        intput_dropout_rate=arguments.intput_dropout_rate,
        weight_decay=arguments.weight_decay,
        ffn_dim=arguments.ffn_dim,
        dataset_name=data_module.dataset_name,
        warmup_updates=arguments.warmup_updates,
        tot_updates=arguments.tot_updates,
        peak_lr=arguments.peak_lr,
        end_lr=arguments.end_lr,
        edge_type=arguments.edge_type,
        multi_hop_max_dist=arguments.multi_hop_max_dist,
        dataset_root=arguments.dataset_root,
        flag=arguments.flag,
        flag_m=arguments.flag_m,
        flag_step_size=arguments.flag_step_size,
        flag_mag=arguments.flag_mag,
        dirpath=arguments.default_root_dir,
    )
    if arguments.checkpoint_path:
        return PathwayTransformer.load_from_checkpoint(arguments.checkpoint_path, strict=False, **model_init_args)
    return PathwayTransformer(**model_init_args)


def run() -> None:
    parser = create_argument_parser()
    arguments = parser.parse_args()
    configure_runtime()
    torch.set_float32_matmul_precision("high")
    if arguments.max_steps < 0:
        arguments.max_steps = arguments.tot_updates + 1

    run_mode = "test" if arguments.test else "validation" if arguments.validate else "training"
    pl.seed_everything(arguments.seed, workers=True, verbose=False)

    graph_data_module = GraphDataModule(
        dataset_name=arguments.dataset_name,
        dataset_root=arguments.dataset_root,
        num_workers=arguments.num_workers,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        multi_hop_max_dist=arguments.multi_hop_max_dist,
        spatial_pos_max=arguments.spatial_pos_max,
    )
    print(f"dataset loading finished for {run_mode}")

    pathway_transformer = create_model(arguments, graph_data_module)
    if arguments.checkpoint_path:
        print(f"checkpoint loaded for {run_mode}: {arguments.checkpoint_path}")
    print(f"model loading finished for {run_mode}")
    if not arguments.test and not arguments.validate:
        print(f"total params: {sum(parameter.numel() for parameter in pathway_transformer.parameters())}")

    checkpoint_dir = os.path.join(arguments.default_root_dir, "lightning_logs", "checkpoints")
    callback_list = [EpochMetricsPrinter()]
    if arguments.enable_early_stopping:
        best_loss_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename="best-val-loss",
            auto_insert_metric_name=False,
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        callback_list.insert(0, best_loss_checkpoint)
        callback_list.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=arguments.early_stopping_patience,
            )
        )
    else:
        last_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=0,
            save_last=True,
        )
        callback_list.insert(0, last_checkpoint)
    lightning_trainer = pl.Trainer(**create_trainer_kwargs(arguments, callback_list))

    resume_checkpoint_path = None
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    if not arguments.test and not arguments.validate and os.path.exists(last_checkpoint_path):
        resume_checkpoint_path = last_checkpoint_path

    if arguments.test:
        test_result = lightning_trainer.test(pathway_transformer, datamodule=graph_data_module)
        if test_result:
            print(f"test_auroc: {test_result[0].get('test_auroc')}")
    elif arguments.validate:
        validation_result = lightning_trainer.validate(pathway_transformer, datamodule=graph_data_module)
        if validation_result:
            print(f"valid_auroc: {validation_result[0].get('valid_auroc')}")
    else:
        lightning_trainer.fit(pathway_transformer, datamodule=graph_data_module, ckpt_path=resume_checkpoint_path)


if __name__ == "__main__":
    run()
