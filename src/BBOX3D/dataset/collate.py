def collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        batch_dict[key] = [sample[key] for sample in batch]
    return batch_dict