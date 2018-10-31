import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RNNDataset, create_dataset
from model import SimpleRNN
from evaluate import generate_predictions
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.01
BATCH_SIZE = 100
NUM_EPOCHS = 100
SEQUENCE_LENGTH = 50
RNN_TYPE = 'GRU'  # Either 'RNN' or 'GRU'


def train_model(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    loss_all = []

    # Train loop.
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            h_state = torch.zeros([model.num_layers, x_batch.size()[0], model.hidden_size]).to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Run our chosen rnn model.
            output, _ = model(x_batch, h_state)

            # Calculate loss.
            loss = loss_function(output, y_batch)

            # Backprop and perform update step.
            loss.backward()
            optimizer.step()

        loss_all.append(loss.cpu().data.numpy())
        print('train loss epoch{}: '.format(epoch), loss.cpu().data.numpy())

    torch.save(model.state_dict(), 'trained_rnn_model.pt')


def main():
    train_x, train_y, test_x, test_y = create_dataset(sequence_length=SEQUENCE_LENGTH, train_percent=0.8)

    train_dataset = RNNDataset(train_x, train_y)
    train_dataloder = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = RNNDataset(test_x, test_y)
    val_dataloader = DataLoader(val_dataset, batch_size=50)

    # Define the model, optimizer and loss function.
    rnn = SimpleRNN(RNN_TYPE, input_size=1, hidden_size=4, num_layers=1).to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_model(rnn, dataloader=train_dataloder, loss_function=loss_function, optimizer=optimizer, epochs=NUM_EPOCHS)

    # Use trained model to make predictions based on an initial sequence of points.
    generate_predictions(rnn, val_dataloader, init_sequence_length=SEQUENCE_LENGTH)


if __name__ == '__main__':
    main()
