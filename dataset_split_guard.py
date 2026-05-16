from pathlib import Path


def resolve_dataset_local_split_csvs(
    dataset_root,
    train_csv,
    val_csv,
    *,
    allow_mismatch=False,
    train_flag='--train-labels-csv',
    val_flag='--val-labels-csv',
):
    """Resolve dataset-local train/val split CSVs when present.

    When a dataset root contains train_split.csv and val_split.csv, prefer that
    fixed protocol automatically unless the caller explicitly overrides it with
    allow_mismatch=True.
    """
    root = Path(dataset_root)
    local_train = root / 'train_split.csv'
    local_val = root / 'val_split.csv'
    messages = []

    if not (local_train.exists() and local_val.exists()):
        return train_csv, val_csv, messages

    def _matches_local(arg, local_path):
        if arg is None:
            return False
        raw = Path(arg).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(root / raw)
        local_resolved = local_path.resolve()
        for cand in candidates:
            try:
                if cand.resolve() == local_resolved:
                    return True
            except FileNotFoundError:
                continue
        return False

    train_matches = _matches_local(train_csv, local_train)
    val_matches = _matches_local(val_csv, local_val)

    if train_csv is None and val_csv is None:
        messages.append(
            f"Detected dataset-local split CSVs in {root}; using {local_train.name} and {local_val.name}."
        )
        return str(local_train), str(local_val), messages

    if train_matches and val_csv is None:
        messages.append(
            f"{train_flag} matches the dataset-local train split; auto-filling {val_flag}={local_val}."
        )
        return str(local_train), str(local_val), messages

    if val_matches and train_csv is None:
        messages.append(
            f"{val_flag} matches the dataset-local val split; auto-filling {train_flag}={local_train}."
        )
        return str(local_train), str(local_val), messages

    if train_matches and val_matches:
        return str(local_train), str(local_val), messages

    mismatch_bits = []
    if not train_matches:
        mismatch_bits.append(f"{train_flag}={train_csv!r}")
    if not val_matches:
        mismatch_bits.append(f"{val_flag}={val_csv!r}")

    message = (
        f"Dataset {root} provides dataset-local split CSVs ({local_train.name}, {local_val.name}), "
        f"but received {', '.join(mismatch_bits)}. This changes the train/validation protocol and "
        f"can break reproducibility. Use those files or pass --allow-split-mismatch to override."
    )
    if allow_mismatch:
        messages.append(f"Warning: {message}")
        return train_csv, val_csv, messages

    raise ValueError(message)