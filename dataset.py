import numpy as np
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    """PyTorch dataset class so we can use their Dataloaders."""

    def __init__(self, x, y=None):
        self.data = x
        self.labels = y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]


def create_dataset(sequence_length, train_percent=0.8):
    """
    Prepare sinosoidal dataset for input into model.
    Data should be in this order: [batch, sequence, feature] where feature is size INPUT_SIZE
    Target labels will the be next value after a sequence of values.
    """

    # Create sin wave at discrete time steps.
    num_time_steps = 1500
    time_steps = np.linspace(start=0, stop=1000, num=num_time_steps, dtype=np.float32)
    discrete_sin_wave = (np.sin(time_steps * 0.5)).reshape(-1, 1)

    # Take (sequence_length + 1) elements & put as a row in sequence_data, extra element is value we want to predict.
    # Move one time step and keep grabbing till we reach the end of our sampled sin wave.
    sequence_data = []
    for i in range(num_time_steps - sequence_length):
        sequence_data.append(discrete_sin_wave[i: i + sequence_length + 1, 0])
    sequence_data = np.array(sequence_data)

    # Split for train/val.
    num_total_samples = sequence_data.shape[0]
    num_train_samples = int(train_percent * num_total_samples)

    train_set = sequence_data[:num_train_samples, :]
    test_set = sequence_data[num_train_samples:, :]

    print('{} total sequence samples, {} used for training'.format(num_total_samples, num_train_samples))

    # Take off the last element of each row and this will be our target value to predict.
    x_train = train_set[:, :-1][:, :, np.newaxis]
    y_train = train_set[:, -1][:, np.newaxis]
    x_test = test_set[:, :-1][:, :, np.newaxis]
    y_test = test_set[:, -1][:, np.newaxis]

    return x_train, y_train, x_test, y_test
