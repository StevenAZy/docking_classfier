import argparse


def add_args(parser):
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--protein_dim1', type=int, default=1280)
    parser.add_argument('--protein_dim2', type=int, default=512)
    parser.add_argument('--protein_dim3', type=int, default=256)
    parser.add_argument('--molecule_dim1', type=int, default=256)
    parser.add_argument('--molecule_dim2', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_dim2', type=int, default=64)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--task_lr', type=float, default=1e-4)
    parser.add_argument('--few_lr', type=float, default=0.01)
    parser.add_argument('--total_epoch', type=int, default=500)
    parser.add_argument('--few_epoch', type=int, default=10)
    parser.add_argument('--num_inner_steps', type=int, default=5)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--explanation', action='store_true', default=False)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--k_query', type=int, default=50)
    parser.add_argument('--val_shot', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--project_name', type=str, default="GCN_maml")
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="")
    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='docking_classfier')
    args = add_args(parser)