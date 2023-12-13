import os
import datetime
import argparse
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import GCNRNADataModule
from GCNModel import GCN_DTIMAML


directory = ["./train_roc_curve_review_v9/", "./zero_roc_curve_review_v9/"]


def add_args(parser):
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--protein_dim1", type=int, default=1280)
    parser.add_argument("--protein_dim2", type=int, default=512)
    parser.add_argument("--protein_dim3", type=int, default=256)
    parser.add_argument("--molecule_dim1", type=int, default=256)
    parser.add_argument("--molecule_dim2", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_dim2", type=int, default=64)
    parser.add_argument("--attention_dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--meta_lr", type=float, default=1e-5)
    parser.add_argument("--task_lr", type=float, default=1e-4)
    parser.add_argument("--few_lr", type=float, default=0.01)
    parser.add_argument("--total_epoch", type=int, default=500)
    parser.add_argument("--few_epoch", type=int, default=10)
    parser.add_argument("--num_inner_steps", type=int, default=5)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--explanation", action="store_true", default=False)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--k_query", type=int, default=50)
    parser.add_argument("--val_shot", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--project_name", type=str, default="GCN_maml")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--checkpoint_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="docking_classfier")
    args = add_args(parser)

    pl.seed_everything(42)

    RNA_data = GCNRNADataModule(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        k_shot=args.k_shot,
        k_query=args.k_query,
        val_shot=args.val_shot,
        test=args.test,
        explanation=args.explanation,
    )

    if not args.test and not args.explanation:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(time)
        for dir in directory:
            if not os.path.exists(dir):
                os.mkdir(dir)
            else:
                for root, dirs, files in os.walk(dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))  # 删除文件
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
        # wandb_logger = WandbLogger(name=args.project_name, project="GCN_maml")
        args.iteration = RNA_data.iterations
        model = GCN_DTIMAML(args=args)

        dirpath = args.project_name + "+checkpoint"
        checkpoint_callback = ModelCheckpoint(
            monitor="zero_auroc",
            dirpath=dirpath,
            filename="-{epoch:03d}-{zero_auroc:.4f}-{zero_loss:.4f}-",
            save_top_k=50,
            mode="max",
            save_last=True,
        )
        # trainer = pl.Trainer(devices=[0],accelerator="gpu",logger=wandb_logger,max_epochs=args.total_epoch,callbacks=[checkpoint_callback]
        #                  ,log_every_n_steps=1)
        trainer = pl.Trainer(
            devices=[0],
            accelerator="gpu",
            max_epochs=args.total_epoch,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

        trainer.callbacks.append(LearningRateMonitor(logging_interval="step"))

    else:
        args.iteration = 1
        molecule_model = GCN_DTIMAML.load_from_checkpoint(
            args.checkpoint_path, args=args
        )
        trainer = pl.Trainer(devices=[0], accelerator="gpu")
    # trainer.validate(molecule_model, datamodule=molecule_data)
    if args.test or args.explanation:
        trainer.test(model, datamodule=RNA_data)

    elif args.val:
        trainer.validate(model, datamodule=RNA_data)
    else:
        trainer.fit(model, datamodule=RNA_data)
