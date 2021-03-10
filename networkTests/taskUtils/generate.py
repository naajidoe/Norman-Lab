import torch # pytorch
import numpy as np  # numpy

"""
Generates a data sample (i.e. matrix) filled with n (n = seqlen1 + seqlen2 + 2) 
vectors of size input_size. The sample contains 2 non-distraction vectors (i.e. 
"input vectors") located at input1_location and input2_location and the rest of
the sample contains "distraction vectors" (i.e. random vectors filled with
numbers between 0 and 1) located everywhere else. Depending on the sequence
type, the generated sample will have equivalent or different non-distraction
vectors
"""
def generate_sample(sequence_type, input_size, seqlen1, seqlen2, seqlen3):
    sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
    X_sample = torch.zeros(sequence_length, input_size)
    Y_sample = 0
    X0 = torch.tensor(np.eye(input_size)[0])
    X1 = torch.tensor(np.eye(input_size)[1])

    # input vectors
    if sequence_type == 0:
        X_sample[seqlen1] = X0
        X_sample[seqlen1 + seqlen2 + 1] = X0
        Y_sample = [0]
    if sequence_type == 1:
        X_sample[seqlen1] = X0
        X_sample[seqlen1 + seqlen2 + 1] = X1
        Y_sample = [1]
    if sequence_type == 2:
        X_sample[seqlen1] = X1
        X_sample[seqlen1 + seqlen2 + 1] = X1
        Y_sample = [0]
    if sequence_type == 3:
        X_sample[seqlen1] = X1
        X_sample[seqlen1 + seqlen2 + 1] = X0
        Y_sample = [1]

    # distraction vectors
    for i in range(sequence_length):
        if i != seqlen1 and i != (seqlen1 + seqlen2 + 1):
            X_sample[i] = torch.rand(input_size)

    return X_sample, Y_sample

"""
Generates a dataset of 4 different versions of the XOR problem based on the
matricies made with the generate_sample function. This dataset will contain
the equivalent of matricies corresponding to the [0, 0, 0], [0, 1, 1], 
[1, 1, 0], and [1, 0, 1] examples of the XOR problem in that order
"""
def generate_dataset(same_distractions, input_size, seqlen1, seqlen2, seqlen3):
    sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
    dataset = torch.zeros(4, sequence_length, input_size)
    dataset[0], YA = generate_sample(0, input_size, seqlen1, seqlen2, seqlen3)
    dataset[1], YB = generate_sample(1, input_size, seqlen1, seqlen2, seqlen3)
    dataset[2], YA = generate_sample(2, input_size, seqlen1, seqlen2, seqlen3)
    dataset[3], YB = generate_sample(3, input_size, seqlen1, seqlen2, seqlen3)

    # when true sets all dataset samples to the have same distraction vectors
    if same_distractions:
        for i in range(sequence_length):
            if i != seqlen1 and i != (seqlen1 + seqlen2 + 1):
                dataset[1][i] = dataset[0][i]
                dataset[2][i] = dataset[0][i]
                dataset[3][i] = dataset[0][i]

    targets = torch.tensor([YA, YB, YA, YB])
    return dataset, targets, sequence_length