import torch
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_predictions(model, dataloader, init_sequence_length):
    """From a trained model predict """
    model.eval()

    h_state = torch.zeros([model.num_layers, 1, model.hidden_size]).to(device)  # Initial state is all zero.
    initial_input = next(iter(dataloader))[1].to(device)  # Grab one initial sequence of data for use in prediction.
    initial_input.data.unsqueeze_(0)  # Need to add our batch dimension back in.

    final_outputs = []
    for _ in range(len(dataloader.dataset.labels)-init_sequence_length):

        output, _ = model(initial_input, h_state)
        final_outputs.append(output.cpu().data.squeeze_())

        # Pop off the first element of sequence then add on our latest generated point (use our predicted values in next predictions).
        initial_input.data[:, 0:init_sequence_length-1, :] = initial_input.data[:, 1:init_sequence_length, :]
        initial_input.data[:, init_sequence_length-1, :] = output.data

    plt.plot(final_outputs, label='predicted')
    plt.plot(dataloader.dataset.labels[init_sequence_length:], label='actual')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    plt.savefig('sin_wave.png')

    return final_outputs
