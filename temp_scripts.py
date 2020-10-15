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
    # device = torch.device("cuda:0")
    model = DETR().eval()
    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": 0.00001,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(param_dicts, lr=0.01,
    #                               weight_decay=0.0001)
    for input_tensor, path in dataloader:
        out = model(input_tensor)
        # print(out)
        break


if __name__ == '__main__':
    demo01()
