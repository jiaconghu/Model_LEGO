from tkinter import *
import numpy as np
import os

FIG_W = 1536
FIG_H = 1024

CONV_W = 4
CONV_H = 4
LINEAR_W = 2
LINEAR_H = 2

INTERVAL_CONV_X = 200
INTERVAL_CONV_Y = 7
INTERVAL_LINEAR_X = 280
INTERVAL_LINEAR_Y = 4.5

PADDING_X = 10
PADDING_Y = 400  # middle line

LINE_WIDTH = 1

# COLOR_PUBLIC = 'orange'
# COLOR_NO_USE = 'gray'
# COLORS = ['purple', 'red']
# COLOR_PUBLIC = '#feb888'
# COLOR_NO_USE = '#c8c8c8'
# COLORS = ['#b0d994', '#a3cbef', ]
COLOR_PUBLIC = '#F8AC8C'
COLOR_NO_USE = '#c8c8c8'
COLORS = ['#C82423', '#2878B5', ]
# COLORS = ['#2878B5', '#C82423', ]


def draw_route(masks, layers):
    root = Tk()
    cv = Canvas(root, background='white', width=FIG_W, height=FIG_H)
    cv.pack(fill=BOTH, expand=YES)

    #  ---------------------------
    #  each layer
    #  ---------------------------
    masks = np.asarray(masks)  # layers, labels, channels
    print(masks.shape)

    x = PADDING_X
    line_start_p_preceding = [(PADDING_X, PADDING_Y)]  # public
    line_start_preceding = [[(PADDING_X, PADDING_Y)] for _ in range(masks.shape[1])]  # [labels * [init]]

    for layer in range(masks.shape[0]):

        line_end_p = []  # public
        line_start_p = []  # public
        line_end = [[] for _ in range(masks.shape[1])]  # [labels * []] each class
        line_start = [[] for _ in range(masks.shape[1])]

        line_p_num = 0
        line_num = 0

        #  ---------------------------
        #  each channel
        #  ---------------------------
        layer_masks = np.asarray(list(masks[layer]))  # labels, channels

        # init posi.
        if layers[layer] == 'conv':
            x += CONV_W + INTERVAL_CONV_X
            y = PADDING_Y - (layer_masks.shape[1] / 2) * (CONV_H + INTERVAL_CONV_Y) + INTERVAL_CONV_Y / 2
        else:
            x += LINEAR_W + INTERVAL_LINEAR_X
            y = PADDING_Y - (layer_masks.shape[1] / 2) * (LINEAR_H + INTERVAL_LINEAR_Y) + INTERVAL_LINEAR_Y / 2

        # draw conv/linear
        for channel in range(layer_masks.shape[1]):
            if layer_masks[:, channel].sum() > 1:
                if layers[layer] == 'conv':
                    line_end_p.append(((x), (y + CONV_H / 2)))
                    line_start_p.append(((x + CONV_W), (y + CONV_H / 2)))
                    cv.create_rectangle(x, y, x + CONV_W, y + CONV_H,
                                        outline=COLOR_PUBLIC,
                                        fill=COLOR_PUBLIC,
                                        width=LINE_WIDTH)
                else:
                    line_end_p.append(((x), (y + LINEAR_H / 2)))
                    line_start_p.append(((x + LINEAR_W), (y + LINEAR_H / 2)))
                    cv.create_oval(x, y, x + LINEAR_W, y + LINEAR_H,
                                   outline=COLOR_PUBLIC,
                                   fill=COLOR_PUBLIC,
                                   width=LINE_WIDTH)
            elif layer_masks[:, channel].sum() < 1:
                if layers[layer] == 'conv':
                    cv.create_rectangle(x, y, x + CONV_W, y + CONV_H,
                                        outline=COLOR_NO_USE,
                                        fill=COLOR_NO_USE,
                                        width=LINE_WIDTH)
                else:
                    cv.create_oval(x, y, x + LINEAR_W, y + LINEAR_H,
                                   outline=COLOR_NO_USE,
                                   fill=COLOR_NO_USE,
                                   width=LINE_WIDTH)
            else:
                #  ---------------------------
                #  each label
                #  ---------------------------
                for l, mask in enumerate(layer_masks[:, channel]):
                    if mask:
                        if layers[layer] == 'conv':
                            line_end[l].append(((x), (y + CONV_H / 2)))
                            line_start[l].append(((x + CONV_W), (y + CONV_H / 2)))
                            cv.create_rectangle(x, y, x + CONV_W, y + CONV_H,
                                                outline=COLORS[l],
                                                fill=COLORS[l],
                                                width=LINE_WIDTH)
                        else:
                            line_end[l].append(((x), (y + LINEAR_H / 2)))
                            line_start[l].append(((x + LINEAR_W), (y + LINEAR_H / 2)))
                            cv.create_oval(x, y, x + LINEAR_W, y + LINEAR_H,
                                           outline=COLORS[l],
                                           fill=COLORS[l],
                                           width=LINE_WIDTH)

            # next y start posi.
            if layers[layer] == 'conv':
                y += CONV_H + INTERVAL_CONV_Y
            else:
                y += LINEAR_H + INTERVAL_LINEAR_Y

        # draw line
        for l in range(layer_masks.shape[0]):
            # line_num += (len(line_start_preceding[l]) * len(line_end[l]))  # each to each
            # line_p_num += (len(line_start_preceding[l]) * len(line_end_p))  # each to public
            # line_p_num += (len(line_start_p_preceding) * len(line_end[l]))  # public to each
            line_num += len(line_start[l])  # each
            for x0, y0 in line_start_preceding[l]:
                # each to each
                for x1, y1 in line_end[l]:
                    cv.create_line(x0, y0, x1, y1,
                                   width=LINE_WIDTH,
                                   fill=COLORS[l],
                                   # arrow=LAST,
                                   arrowshape=(6, 5, 1))

                # each to public
                for x1, y1 in line_end_p:
                    cv.create_line(x0, y0, x1, y1,
                                   width=LINE_WIDTH,
                                   fill=COLORS[l],
                                   # arrow=LAST,
                                   arrowshape=(6, 5, 1))

            # public to each
            for x0, y0 in line_start_p_preceding:
                for x1, y1 in line_end[l]:
                    cv.create_line(x0, y0, x1, y1,
                                   width=LINE_WIDTH,
                                   fill=COLORS[l],
                                   # arrow=LAST,
                                   arrowshape=(6, 5, 1))

        # line_p_num += (len(line_start_p_preceding) * len(line_end_p))  # public to public
        line_p_num += len(line_start_p)  # public
        # public to public
        for x0, y0 in line_start_p_preceding:
            for x1, y1 in line_end_p:
                cv.create_line(x0, y0, x1, y1,
                               width=LINE_WIDTH + 1,
                               fill=COLOR_PUBLIC,
                               # arrow=LAST,
                               arrowshape=(6, 5, 1))

        line_start_preceding = line_start.copy()
        line_start_p_preceding = line_start_p.copy()

        # calculate
        print('--->', layer,
              '| line--->', line_num,
              '| line_p--->', line_p_num,
              '| --->', line_p_num / (line_num + line_p_num))

    root.mainloop()


def load_mask():
    labels = [2]
    layers = [3, 2, 1, 0]  # inputs
    layers_name = ['conv' for _ in range(2)] + ['linear' for _ in range(3)]

    grads_path = os.path.join(r'/nfs3-p1/hjc/cnnlego/output/vgg16_cifar10_10111427/locating/locating_layer{}.npy')

    masks = []
    for layer in layers:
        layer_masks = []
        for label in labels:
            output_channel = np.load(grads_path.format(layer))[label]
            layer_masks.append(output_channel)
        masks.append(layer_masks)

    layer_masks = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    masks.append(layer_masks)
    return masks, layers_name


def main():
    masks, layers = load_mask()
    print(np.asarray(masks).shape)
    print(layers)

    draw_route(masks, layers)


if __name__ == '__main__':
    main()
