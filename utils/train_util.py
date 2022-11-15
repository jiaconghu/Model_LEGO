class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}[VAL:{val' + self.fmt + '} AVG:{avg' + self.fmt + '}]'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total, step, prefix, meters):
        self._fmtstr = self._get_fmtstr(total)
        self.meters = meters
        self.prefix = prefix

        self.step = step

    def display(self, running):
        if running % self.step == 0:
            entries = [self.prefix + self._fmtstr.format(running)]  # [prefix xx.xx/xx.xx]
            entries += [str(meter) for meter in self.meters]
            print('  '.join(entries))

    def _get_fmtstr(self, total):
        num_digits = len(str(total // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total) + ']'  # [prefix xx.xx/xx.xx]
