import torch
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from nets.detr import DETR


def demo01():
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=False,
                           augments=True,
                           debug=60,
                           min_thresh=400,
                           max_thresh=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    device = torch.device("cuda:0")
    model = DETR().to(device)
    for input_tensor, path in dataloader:
        # print(input_tensor.boxes)
        out = model(input_tensor.to(device))
        # print(out)
        break


if __name__ == '__main__':
    demo01()
