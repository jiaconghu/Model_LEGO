import torch


def accuracy(outputs, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)  # [batch_size, topk]
        pred = pred.t()  # [topk, batch_size]
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # [topk, batch_size]

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ClassAccuracy:
    def __init__(self):
        self.sum = {}
        self.count = {}

    def accuracy(self, outputs, labels):
        _, pred = outputs.max(dim=1)
        correct = pred.eq(labels)

        for b, label in enumerate(labels):
            label = label.item()
            if label not in self.sum.keys():
                self.sum[label] = 0
                self.count[label] = 0
            self.sum[label] += correct[b]
            self.count[label] += 1

    def __call__(self):
        self.sum = dict(sorted(self.sum.items()))
        self.count = dict(sorted(self.count.items()))
        return [s / c * 100 for s, c in zip(self.sum.values(), self.count.values())]

    def __getitem__(self, item):
        return self.__call__()[item]

    def __str__(self):
        fmtstr = '{}:{:6.2f}'
        result = '\n'.join([fmtstr.format(l, a) for l, a in enumerate(self.__call__())])
        return result
