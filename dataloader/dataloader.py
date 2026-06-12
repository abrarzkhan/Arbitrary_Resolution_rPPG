import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class MHDataLoader(DataLoader):
    """
    Unchanged interface. Note: with the in-RAM frame cache in PURE_train, the
    recommended setup (especially on Windows, which uses 'spawn' and would copy
    the cache into every worker) is n_threads=0 so the cache lives once in the
    main process and is reused across epochs.
    """
    def __init__(self, args, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None,
                 collate_fn=default_collate, pin_memory=True, drop_last=False,
                 timeout=0, worker_init_fn=None):
        num_workers = args.n_threads if hasattr(args, 'n_threads') and args.n_threads > 0 else 0
        super(MHDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=(num_workers > 0),
        )
        self.scale = args.scale