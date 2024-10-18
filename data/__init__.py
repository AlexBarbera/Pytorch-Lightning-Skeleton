import torch
import MyDataset


def get_dataloader(path, **kwargs):
    # TODO implement dataset
    data = MyDataset()

    # TODO add extra params to dataloader

    return torch.utils.data.DataLoader(data, **kwargs)
