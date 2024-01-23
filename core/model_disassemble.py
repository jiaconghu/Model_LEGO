import os
import argparse

import torch
import torch_pruning as tp

import models


def disassemble():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    parser.add_argument('--mask_dir', default='', type=str, help='mask dir')
    parser.add_argument('--disa_layers', default='', nargs='+', type=int, help='disa layers')
    parser.add_argument('--disa_labels', default='', nargs='+', type=int, help='disa labels')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    print('-' * 50)
    print('SAVE DIR:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # model configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    # model = torch.load(args.model_path).cpu()

    modules = models.load_modules(model=model, model_layers=None)

    # ----------------------------------------
    # disa configuration
    # ----------------------------------------
    mask_path = os.path.join(args.mask_dir, 'mask_layer{}.pt')

    if args.disa_layers[0] == -1:
        args.disa_layers = [i for i in range(len(modules) - 1)]

    print('disassembling layers:', args.disa_layers)
    print('disassembling labels:', args.disa_labels)

    # ----------------------------------------
    # model disassemble
    # ----------------------------------------
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))

    ###############################
    # layers 1-N: input channels
    ###############################
    for layer in args.disa_layers:
        print('===> LAYER', layer)
        print('--->', modules[layer])

        # idxs
        mask_total_i = None
        mask_i = torch.load(mask_path.format(layer))
        for label in args.disa_labels:
            if mask_total_i is None:
                mask_total_i = mask_i[label]
            else:
                mask_total_i = torch.bitwise_or(mask_i[label], mask_total_i)
        idxs = torch.where(mask_total_i == 0)[0].tolist()

        # structure pruning
        prune_fn = None
        if isinstance(modules[layer], torch.nn.Conv2d):
            prune_fn = tp.prune_conv_in_channels
        if isinstance(modules[layer], torch.nn.Linear):
            prune_fn = tp.prune_linear_in_channels
        group = DG.get_pruning_group(modules[layer], prune_fn, idxs=idxs)
        if DG.check_pruning_group(group):
            group.prune()
        print('--->', modules[layer])

    ###############################
    # layer N: output channels
    ###############################
    # layer = 0
    # print('--->', modules[layer])
    #
    # # idxs
    # mask_i = torch.load(mask_path.format(-1))
    # mask_total_i = None
    # for label in args.disa_labels:
    #     if mask_total_i is None:
    #         mask_total_i = mask_i[label]
    #     else:
    #         mask_total_i = torch.bitwise_or(mask_i[label], mask_total_i)
    # idxs = np.where(mask_total_i == 0)[0].tolist()
    #
    # # structure pruning
    # prune_fn = tp.prune_linear_out_channels
    # group = DG.get_pruning_group(modules[layer], prune_fn, idxs=idxs)
    # if DG.check_pruning_group(group):
    #     group.prune()
    # print('--->', modules[layer])

    ###############################
    # save model
    ###############################
    model.zero_grad()
    result_path = os.path.join(args.save_dir, 'model_disa.pth')
    torch.save(model, result_path)
    print(model)


if __name__ == '__main__':
    disassemble()
