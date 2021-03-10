import torch
import numpy as np  # numpy

# Tests rounded network outputs against correct network outputs based on sample
def test_network(sample_number, dataset, targets, network, input_size, batch_size, sequence_length):
    # Test network
    test_sample = sample_number
    test_data = dataset[test_sample].view(batch_size, sequence_length, 
                                          input_size)
    test_targets = torch.tensor([0.0, 0.0])
    if targets[test_sample] != 0:
        test_targets = torch.tensor([1.0, 0.0])
    else:
        test_targets = torch.tensor([0.0, 1.0])
    test = network(test_data)
    test = torch.softmax(test, dim=1)

    print("\n")
    print("Test of network: ")
    print("input is {}".format(test_data.detach().numpy()))
    print('out is {}'.format(torch.round(test).detach().numpy()))
    print('expected out is {}'.format(test_targets.detach().numpy()))