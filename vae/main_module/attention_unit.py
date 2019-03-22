import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(2 * encoder_hidden_dim + decoder_hidden_dim, 2 * encoder_hidden_dim)
        self.v = nn.Linear(2 * encoder_hidden_dim, 1)

    def forward(self, encoder_out, hidden, lengths):
        '''
        :param encoder_out: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param hidden: [decoder_hidden_dim, batch]
        :param lengths: [batch]
        '''
        hidden = hidden.unsqueeze(2)
        # hidden = [decoder_hidden_dim, batch, 1]
        hidden = hidden.repeat(1, 1, encoder_out.size()[1])
        # hidden = [decoder_hidden_dim, batch, max_seq_len]
        hidden = hidden.permute(1, 2, 0)
        # hidden = [batch, max_seq_len, decoder_hidden_dim]
        temp = torch.cat((hidden, encoder_out), dim=2)
        # temp = [batch, max_seq_len, 2 x encoder_hidden_dim + decoder_hidden_dim]
        energy = torch.zeros(encoder_out.size()[0], encoder_out.size()[1])
        for batch in range(encoder_out.size()[0]):
            energy[batch, :lengths[batch].item()] = F.softmax(self.v(torch.tanh(self.attn(temp[batch, : lengths[batch].item()]))), dim=0).squeeze(1)
            # [seq_len, 2 x encoder_hidden_dim + decoder_hidden_dim] -> [seq_len, 2 x encoder_hidden_dim] -> [seq_len, 1] -> [seq_len]
        # energy = [batch, max_seq_len]
        context = torch.bmm(energy.unsqueeze(1), encoder_out).squeeze(1)
        # context = [batch, 2 x encoder_hidden_dim]
        return context


if __name__ == "__main__":
    attn_model = Attention(3, 4)
    encoder_out = torch.randn(3, 3, 6)
    hidden = torch.randn(4, 3)
    lengths = torch.tensor([2, 1, 1])
    context = attn_model(encoder_out, hidden, lengths)
    print(context)

