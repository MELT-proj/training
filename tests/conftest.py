# nyx enforces strict overcommit accounting (vm.overcommit_memory=2): every
# fork() must reserve commit-ledger space for the *entire* parent address
# space, and every mmap-loaded checkpoint is checked against the same ledger.
# pytest is one long-lived process, so its footprint grows across the run as
# more modules/models get imported/loaded -- tests that fork DataLoader
# workers or mmap-load checkpoints get more expensive to run the later they
# run. Front-load those files so they run against the smallest possible
# process footprint instead of whatever's accumulated by the time they'd
# normally execute.
_FORK_HEAVY_FILES = (
    "test_check_training_config.py",
    "test_lhotse_dataloader.py",
    "test_trainer.py",
)


def pytest_collection_modifyitems(items):
    def rank(item):
        for i, name in enumerate(_FORK_HEAVY_FILES):
            if item.location[0].endswith(name):
                return i
        return len(_FORK_HEAVY_FILES)

    items.sort(key=rank)
