from torchvision import datasets, transforms
import torch.utils.data
from deep_eurorack_control.datasets.nsynth_dataset import NSynthDataset
from deep_eurorack_control.config import settings


def mnist_data_loader(batch_size, data_dir, valid_ratio=0.2, num_threads=0):
    # Load the dataset for the training/validation sets
    train_valid_set = datasets.MNIST(
        root=data_dir, train=True, transform=transforms.ToTensor(), download=True
    )

    # Split it into training and validation sets
    # if valid_ratio = 0.2 : 80%/20% split for train/valid
    nb_train = int((1.0 - valid_ratio) * len(train_valid_set))
    nb_valid = int(valid_ratio * len(train_valid_set))
    train_set, valid_set = torch.utils.data.dataset.random_split(train_valid_set, [nb_train, nb_valid])

    # Load the test set
    test_set = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())

    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    return train_loader, valid_loader, test_loader


def nsynth_data_loader(batch_size, data_dir=settings.DATA_DIR, audio_dir=settings.AUDIO_DIR, nsynth_json="nsynth_string.json", valid_ratio=0.2, num_threads=0):
    # Load the dataset for the training/validation sets
    train_valid_set = NSynthDataset(
        data_dir=data_dir,
        audio_dir=audio_dir,
        nsynth_json=nsynth_json,
        transform=None  # transforms.ToTensor()
    )

    # Split it into training and validation sets
    # if valid_ratio = 0.2 : 80%/20% split for train/valid
    nb_train = round((1.0 - valid_ratio) * len(train_valid_set))
    nb_valid = round(valid_ratio * len(train_valid_set))
    train_set, valid_set = torch.utils.data.dataset.random_split(train_valid_set, [nb_train, nb_valid])

    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_threads)
    return train_loader, valid_loader
