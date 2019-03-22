import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, hidden, encoder_outs, lengths):
        '''
        :param input: [batch,
        :param hidden:
        :param encoder_outs:
        :param lengths:
        '''