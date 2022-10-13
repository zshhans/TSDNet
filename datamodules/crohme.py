import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
from collections import OrderedDict
from typing import List
from .utils import Vocabulary, Collate, AdaptiveBatchSampler


class CROHMEDataset(Dataset):
    def __init__(self,
                 dir_name,
                 node_vocab: Vocabulary,
                 edge_vocab: Vocabulary,
                 indices=None,
                 transform=transforms.ToTensor()):
        super().__init__()
        self.dir_name = dir_name
        self.root_dir = os.path.join("data/crohme", dir_name)
        df = pd.read_csv(
            os.path.join(self.root_dir, "metadata.txt"),
            delimiter="\t",
            header=0,
        )
        self.metadata = df[df["latex"].apply(lambda x: False if "\\sqrt [" in x
                                             else True)].reset_index(drop=True)
        if indices is not None:
            self.metadata = self.metadata.loc[indices].reset_index(drop=True)
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample_id = self.metadata.iloc[index, 0]
        img_path = os.path.join(self.root_dir, "bmp", f"{sample_id}.bmp")
        img = Image.open(img_path)
        img = self.transform(img)

        lg_path = os.path.join(self.root_dir, "tree", f"{sample_id}.lg")
        nodes, edges = self._read_lg_file(lg_path)

        return img, nodes, edges, sample_id

    def _read_lg_file(self, lg_path):
        objs = []
        rels_dict = OrderedDict()

        with open(lg_path, "r") as fin:
            for line in fin.readlines():
                tokens = line[:-1].split(", ")
                if line.startswith("O"):
                    objs.append({
                        "id": tokens[1],
                        "type": tokens[2],
                        "path": tokens[4]
                    })
                elif line.startswith("R"):
                    if (tokens[2]) in rels_dict:
                        if tokens[3] == "Inside":
                            continue
                    rels_dict[tokens[2]] = {
                        "src": tokens[1],
                        "dst": tokens[2],
                        "type": tokens[3],
                    }
        rels = rels_dict.values()

        objs = pd.DataFrame(objs, columns=["id", "type", "path"])
        sos_obj_id = f"{Vocabulary.sos_tok}_0"
        eos_obj_id = f"{Vocabulary.eos_tok}_1024"
        objs = objs.append(
            {
                "id": sos_obj_id,
                "type": Vocabulary.sos_tok,
                "path": ""
            },
            ignore_index=True,
        )
        objs = objs.append(
            {
                "id": eos_obj_id,
                "type": Vocabulary.eos_tok,
                "path": ""
            },
            ignore_index=True,
        )
        objs = (objs.sort_values(
            by="id",
            key=lambda col: col.apply(lambda x: int(x.split("_")[-1])),
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
            key=lambda col: col.apply(lambda x: int(x.split("_")[-1])),
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

    def random_split(self, splits: List[float]):
        assert sum(splits) == 1
        indices = torch.randperm(self.__len__()).tolist()
        lengths = [int(self.__len__() * x) for x in splits]
        lengths[-1] = self.__len__() - sum(lengths[:-1])
        subsets = []
        for i, length in enumerate(lengths):
            start = sum(lengths[:i])
            stop = start + lengths[i]
            subsets.append(
                CROHMEDataset(
                    self.dir_name,
                    self.node_vocab,
                    self.edge_vocab,
                    indices=indices[start:stop],
                ))
        return subsets


class CROHMEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        node_vocab: Vocabulary,
        edge_vocab: Vocabulary,
        batch_size=16,
        rand_size=128,
        mem_size=5e5,
        train_dir="train2014",
        val_dir="test2014",
        test_dir="test2014",
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.collate_fn = Collate(node_vocab.get_pad_idx(),
                                  edge_vocab.get_pad_idx())
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.batch_size = batch_size
        self.rand_size = rand_size
        self.mem_size = mem_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            ds = CROHMEDataset(self.train_dir, self.node_vocab,
                               self.edge_vocab)
            if self.val_dir is None:
                self.train_ds, self.val_ds = ds.random_split([0.9, 0.1])
            else:
                self.train_ds = ds
                self.val_ds = CROHMEDataset(self.val_dir, self.node_vocab,
                                            self.edge_vocab)

        if stage == "test" or stage is None:
            self.test_ds = CROHMEDataset(self.test_dir, self.node_vocab,
                                         self.edge_vocab)

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
