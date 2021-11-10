import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attn(nn.Module):

    """ Self attention Layer applied after conv5 layer (final layer) of resnet to learn dependencies between music components """

    def __init__(self, input_channels):
        super(Self_Attn, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
            args : x : input feature maps ---> [batch,n_head,W,H]
            returns : output feature map 
                      attention
        """
        m_batchsize, channels, width, height = input.size()
        proj_query = self.conv1(input).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # (batch,n_head,W*H)->(batch,W*H,n_head)
        proj_key = self.conv2(input).view(m_batchsize, -1, width * height)
        # (batch,n_head,W*H)
        energy = torch.bmm(proj_query, proj_key)
        # transpose check  (batch,W*H,W*H)
        attention = self.softmax(energy)  # (batch,W*H,W*H)
        proj_value = self.conv3(input).view(m_batchsize, -1, width * height)
        # (batch,n_head,W*H)
        output = torch.bmm(proj_value, attention.permute(0, 2, 1))
        #(batch,n_head,W*H) * (batch,W*H,W*H) = (batch,n_head,W*H)
        output = output.view(m_batchsize, channels, width, height)
        #(batch,n_head,W,H)
        output = self.gamma * output + input
        return output, attention

class Attention(nn.Module):
    """
      Attention layer applied after conv5 layer (final layer) of resnet
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.attention_dictionary_size, self.attention_dictionary_size*self.num_heads, bias=False)
        self.fc2 = nn.Linear(self.attention_dictionary_size, self.attention_dictionary_size*self.num_heads, bias=False)
        self.fc3 = nn.Linear(self.attention_dictionary_size, self.attention_dictionary_size*self.num_heads, bias=False)
        self.fc4 = nn.Linear(self.attention_dictionary_size, self.attention_dictionary_size*self.num_heads, bias=False)
        self.num_heads = 10
        self.attention_dictionary_size = 20

    def forward(self, input):
        input = input.unsqueeze(1)
        batch, m, _= input.size()
        query = self.fc1(input)
        query = query.view(self.num_heads, batch, m, self.attention_dictionary_size)
        keys = self.fc2(input).view(self.num_heads, batch, self.attention_dictionary_size, m)
        values = self.fc3(input).view(self.num_heads, batch, m, self.attention_dictionary_size)
        inner_pro = query.matmul(keys)  # num_heads * batch * m * m
        att_score = self.softmax(inner_pro)
        result = (att_score.permute(0, 1, 3, 2)).matmul(values)
        result = result.view(batch, m, -1)  # batch, m * att_emb_size * num_heads
        result = F.relu(self.gamma * result + self.fc4(input))  # batch , m , att_emb_size*num_heads
        return result
