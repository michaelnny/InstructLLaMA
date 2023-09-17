import torch


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = 0

        self.deltas = []
        self.min_size = 10

    @torch.no_grad()
    def update(self, x):
        x = x.to(self.mean.device)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    def normalize(self, x, shift_mean=True):
        if self.var == 0:
            if shift_mean:
                return x - self.mean.to(x.device)
            else:
                return x
        else:
            if shift_mean:
                return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device))
            else:
                return (x) / torch.sqrt(self.var.to(x.device))
