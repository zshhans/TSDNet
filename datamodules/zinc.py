# Torch & Lightning
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Utils
import pandas as pd
import pickle as pkl
from PIL import Image
from pathlib import Path
from collections import OrderedDict

# Dataset Utils
from .utils import Vocabulary, Collate, AdaptiveBatchSampler


class ZINCDataset(Dataset):
    def __init__(self,
                 root_dir,
                 node_vocab,
                 edge_vocab,
                 indices=None,
                 transform=transforms.ToTensor()):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_csv(
            self.root_dir / "metadata.txt",
            delimiter="\t",
            header=0,
        )
        if indices is not None:
            self.metadata = self.metadata.loc[indices].reset_index(drop=True)
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample_id = self.metadata.iloc[index, 0]
        img_path = self.root_dir / "png" / f"{sample_id}.png"
        img = Image.open(img_path)
        img = self.transform(img)

        lg_path = self.root_dir / "tree" / f"{sample_id}.lg"
        nodes, edges = self._read_lg_file(lg_path)

        return img, nodes, edges, sample_id

    def _read_lg_file(self, lg_path):
        objs = []
        rels_dict = OrderedDict()

        with lg_path.open("r") as fin:
            for line in fin.readlines():
                tokens = line[:-1].split(", ")
                if line.startswith("O"):
                    objs.append({"id": tokens[1], "type": tokens[2]})
                elif line.startswith("R"):
                    rels_dict[tokens[2]] = {
                        "src": tokens[1],
                        "dst": tokens[2],
                        "type": tokens[3],
                    }
        rels = rels_dict.values()

        objs = pd.DataFrame(objs, columns=["id", "type"])
        sos_obj_id = "0"
        eos_obj_id = "1024"
        objs = objs.append(
            {
                "id": sos_obj_id,
                "type": Vocabulary.sos_tok
            },
            ignore_index=True,
        )
        objs = objs.append(
            {
                "id": eos_obj_id,
                "type": Vocabulary.eos_tok
            },
            ignore_index=True,
        )
        objs = (objs.sort_values(
            by="id",
            key=lambda col: col.apply(lambda x: int(x)),
            ignore_index=True,
        ).reset_index().set_index("id"))

        rels = pd.DataFrame(rels, columns=["src", "dst", "type"])
        rels = rels.append(
            {
                "src": sos_obj_id,
                "dst": objs.index[1],
                "type": Vocabulary.pad_tok
            },
            ignore_index=True,
        )
        rels = rels.append(
            {
                "src": objs.index[-2],
                "dst": eos_obj_id,
                "type": Vocabulary.pad_tok
            },
            ignore_index=True,
        )
        rels = rels.sort_values(
            by="dst",
            key=lambda col: col.apply(lambda x: int(x)),
            ignore_index=True,
        )

        objs.type = objs.type.apply(lambda x: self.node_vocab.stoi[x])
        rels.type = rels.type.apply(lambda x: self.edge_vocab.stoi[x])
        rels["src_type"] = rels.src.apply(lambda x: objs.loc[x, "type"])
        rels["dst_type"] = rels.dst.apply(lambda x: objs.loc[x, "type"])
        rels.src = rels.src.apply(lambda x: objs.loc[x, "index"])
        rels.dst = rels.dst.apply(lambda x: objs.loc[x, "index"])

        objs = objs.reset_index(drop=True).drop(columns="index")

        return objs, rels


class ZINCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        node_vocab: Vocabulary,
        edge_vocab: Vocabulary,
        data_dir='data/zinc',
        mode="full",
        batch_size=16,
        rand_size=128,
        mem_size=5e5,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.collate_fn = Collate(node_vocab.get_pad_idx(),
                                  edge_vocab.get_pad_idx())
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.batch_size = batch_size
        self.rand_size = rand_size
        self.mem_size = mem_size
        self.mode = mode
        assert mode in ["full", "easy", "mid", "hard"]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if self.mode == "full":
            splits_file = Path(self.data_dir) / "full_splits.pkl"
            with splits_file.open("rb") as f:
                splits = pkl.load(f)
        else:
            splits_file = Path(self.data_dir) / f"{self.mode}_splits.pkl"
            with splits_file.open("rb") as f:
                splits = pkl.load(f)
        self.train_ds = ZINCDataset(self.data_dir,
                                    self.node_vocab,
                                    self.edge_vocab,
                                    indices=splits[0])
        self.val_ds = ZINCDataset(self.data_dir,
                                  self.node_vocab,
                                  self.edge_vocab,
                                  indices=splits[1])
        self.test_ds = ZINCDataset(self.data_dir,
                                   self.node_vocab,
                                   self.edge_vocab,
                                   indices=splits[2])

    def train_dataloader(self):
        batch_sampler = AdaptiveBatchSampler(
            self.train_ds.metadata,
            batch_size=self.batch_size,
            rand_size=self.rand_size,
            mem_size=self.mem_size,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
