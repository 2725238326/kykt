import sys

from tqdm.auto import tqdm


class TqdmBar(object):
    def __init__(self, total, desc, position=0, leave=True):
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.bar = None
        self.current = 0

    def start(self):
        if self.bar is None:
            self.bar = tqdm(
                total=self.total,
                desc=self.desc,
                position=self.position,
                leave=self.leave,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            self.current = 0

    def update(self, value):
        if self.bar is None:
            self.start()
        delta = value - self.current
        if delta > 0:
            self.bar.update(delta)
        self.current = value

    def finish(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class TqdmWriter(object):
    def write(self, string):
        tqdm.write(string, file=sys.stdout)

    def flush(self):
        return


class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size

        self.epoch_bar = TqdmBar(total=n_epochs, desc='Epoch', position=0, leave=True)
        self.train_writer = TqdmWriter()
        self.valid_writer = TqdmWriter()

        self.reset_train_bar()
        self.reset_valid_bar()

    def reset_train_bar(self, train_size=None):
        if hasattr(self, 'train_bar') and self.train_bar is not None:
            self.train_bar.finish()
        total = train_size if train_size is not None else self.train_size
        self.train_bar = TqdmBar(total=total, desc='Train', position=1, leave=False)

    def reset_valid_bar(self):
        if hasattr(self, 'valid_bar') and self.valid_bar is not None:
            self.valid_bar.finish()
        self.valid_bar = TqdmBar(total=self.valid_size, desc='Valid', position=2, leave=False)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert len(val) == self.meters
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
