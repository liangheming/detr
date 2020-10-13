import torch
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from nets.detr import BackBone


def demo01():
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=False,
                           augments=True,
                           debug=60
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    device = torch.device("cuda:0")
    # model = BackBone().to(device)
    model = BackBone()
    for input_tensor, path in dataloader:
        # input_tensor.to(device)
        model(input_tensor)
        break


if __name__ == '__main__':
    demo01()
