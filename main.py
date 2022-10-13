import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules import Vocabulary, CROHMEDataModule, ZINCDataModule
from models import TSDNetModule


def get_args_parser():

    parser = argparse.ArgumentParser(description="TSDNet")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument("--dataset",
                              required=True,
                              choices=['crohme', 'zinc'])
    train_parser.add_argument("--log_dir",
                              default='logs',
                              help="TensorBoard log output directory")
    train_parser.add_argument("--exp_name",
                              default='',
                              help="TensorBoard experiment name")
    train_parser.add_argument(
        "--config",
        default=None,
        help="path to model config file, see \"configs/crohme.yml\" for example"
    )
    train_parser.add_argument("--seed", default=42, type=int)
    train_parser.add_argument("--gpu", default=0, type=int)
    train_parser.add_argument("--progress_bar", action="store_true")

    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)
    test_parser.add_argument("--dataset",
                             required=True,
                             choices=['crohme', 'zinc'])
    test_parser.add_argument(
        "--config",
        default=None,
        help="path to model config file, see \"configs/crohme.yml\" for example"
    )
    test_parser.add_argument(
        "--state_dict",
        default=None,
        help="path to the state_dict of the model to be tested")
    test_parser.add_argument("--gpu", default=0, type=int)

    return parser


def train(args):
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(args.seed)
    if args.dataset == 'crohme':
        node_vocab = Vocabulary("data/crohme/node_dict.txt")
        edge_vocab = Vocabulary("data/crohme/edge_dict.txt",
                                use_sos=False,
                                use_eos=False)
        dm = CROHMEDataModule(node_vocab, edge_vocab)
    else:
        node_vocab = Vocabulary("data/zinc/node_dict.txt")
        edge_vocab = Vocabulary("data/zinc/edge_dict.txt",
                                use_sos=False,
                                use_eos=False)
        dm = ZINCDataModule(node_vocab, edge_vocab)
    dm.setup()
    logger = TensorBoardLogger(args.log_dir, args.exp_name)
    if not args.config:
        if args.dataset == 'crohme':
            config_path = "configs/crohme.yml"
        else:
            config_path = "configs/zinc.yml"
    else:
        config_path = args.config
    config_f = open(config_path, 'r')
    config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
    config_f.close()
    model = TSDNetModule(**config_dict)
    checkpoint_callback = ModelCheckpoint(monitor="val_ExpRate",
                                          save_top_k=2,
                                          mode="max",
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = pl.Trainer(gpus=[args.gpu],
                         callbacks=callbacks,
                         logger=logger,
                         enable_progress_bar=args.progress_bar,
                         max_epochs=-1,
                         gradient_clip_val=1)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


def test(args):
    if args.dataset == 'crohme':
        node_vocab = Vocabulary("data/crohme/node_dict.txt")
        edge_vocab = Vocabulary("data/crohme/edge_dict.txt",
                                use_sos=False,
                                use_eos=False)
        dm14 = CROHMEDataModule(node_vocab, edge_vocab, test_dir="test2014")
        dm16 = CROHMEDataModule(node_vocab, edge_vocab, test_dir="test2016")
        dm19 = CROHMEDataModule(node_vocab, edge_vocab, test_dir="test2019")
        dm14.setup("test")
        dm16.setup("test")
        dm19.setup("test")
        if not args.config:
            config_path = "configs/crohme.yml"
        else:
            config_path = args.config

        config_f = open(config_path, 'r')
        config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
        config_f.close()
        model = TSDNetModule(**config_dict)
        if not args.state_dict:
            state_dict_path = "trained_models/tsdnet_crohme.pt"
        else:
            state_dict_path = args.state_dict
        state_dict = torch.load(state_dict_path,
                                torch.device('cuda', args.gpu))
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            print("Testing on CROHME14:")
            model.beam_test(dm14.test_dataloader(), node_vocab, edge_vocab)
            print("Testing on CROHME16:")
            model.beam_test(dm16.test_dataloader(), node_vocab, edge_vocab)
            print("Testing on CROHME19:")
            model.beam_test(dm19.test_dataloader(), node_vocab, edge_vocab)

    else:
        node_vocab = Vocabulary("data/zinc/node_dict.txt")
        edge_vocab = Vocabulary("data/zinc/edge_dict.txt",
                                use_sos=False,
                                use_eos=False)
        dm = ZINCDataModule(node_vocab, edge_vocab)
        dm.setup()
        if not args.config:
            config_path = "configs/zinc.yml"
        else:
            config_path = args.config

        config_f = open(config_path, 'r')
        config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
        config_f.close()
        model = TSDNetModule(**config_dict)
        if not args.state_dict:
            state_dict_path = "trained_models/tsdnet_zinc.pt"
        else:
            state_dict_path = args.state_dict
        state_dict = torch.load(state_dict_path,
                                torch.device('cuda', args.gpu))
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            print("Testing on ZINC:")
            model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.func(args)
