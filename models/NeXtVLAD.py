from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, dim=1024, num_clusters=64, lamb=2, groups=8, max_frames=300):
        super(NeXtVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)
        # expansion FC
        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        #         print(f"x: {x.shape}")

        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)

        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        alpha_g = torch.sigmoid(self.fc_g(x_dot))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        return vlad

class NeXtVLADModel(nn.Module):
    def __init__(self, config):
        super(NeXtVLADModel, self).__init__()

        self.video_model = NeXtVLAD(config.video_dim, max_frames=config.video_max_frames, lamb=config.lamb,
                                       num_clusters=config.num_clusters, groups=config.groups)

        self.audio_model = NeXtVLAD(config.audio_dim, max_frames=config.audio_max_frames, lamb=config.lamb,
                                       num_clusters=config.num_clusters, groups=config.groups)

        self.text_model = NeXtVLAD(config.text_dim, max_frames=config.text_max_frames *2, lamb=config.lamb,
                                       num_clusters=config.num_clusters, groups=config.groups)
        
        # self.text_model_2 = NeXtVLAD(config.text_dim_2, max_frames=config.text_max_frames, lamb=config.lamb,
        #                                num_clusters=config.num_clusters, groups=config.groups)

        # fusion model
        # line_hidden = config.video_hidden
        # line_hidden = config.video_hidden + config.audio_hidden
        line_hidden = config.video_hidden + config.audio_hidden + config.text_hidden
        # line_hidden = config.video_hidden + config.audio_hidden + config.text_hidden + config.text_hidden_2
        self.fusion_model = SeFusion(config,line_hidden)
        self.classifier = SigmodModel(config.hidden_size,config.num_classes)

    def forward(self,data,device):

        video_feature = data['video_feature'].to(device)
        audio_feature = data['audio_feature'].to(device)
        text_feature = data['text_feature'].to(device)
        text_feature_2 = data['text_feature_2'].to(device)

        # video_feature = video_feature * text_feature
        # video_feature = torch.cat([video_feature,text_feature,text_feature_2],1)
        text_feature = torch.cat([text_feature,text_feature_2],1) 
   
        video_feature = self.video_model(video_feature)
        audio_feature = self.audio_model(audio_feature)
        text_feature = self.text_model(text_feature)
        # text_feature_2 = self.text_model_2(text_feature_2)

        # fusion_feature = self.fusion_model([video_feature,audio_feature])
        fusion_feature = self.fusion_model([video_feature,audio_feature,text_feature])
        # fusion_feature = self.fusion_model([video_feature,audio_feature,text_feature,text_feature_2])
        logits,output = self.classifier(fusion_feature)

        return logits,output 

class SeFusion(nn.Module):
    pass
    def __init__(self,line_hidden):
        super(SeFusion,self).__init__()
        self.drop_rate = 0.5
        gating_reduction = 4
        hidden_size = 2048
        # self.fc0 = nn.Linear(config.text_hidden + config.video_hidden + config.audio_hidden,  config.hidden_size)
        self.fc0 = nn.Linear(line_hidden,  hidden_size)
        self.bn0 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // gating_reduction)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(hidden_size // gating_reduction, hidden_size)
        self.bn2 = nn.BatchNorm1d(1)


    def forward(self,feature_list):
        # B x (f1+f2+ ...)
        con_feature = torch.cat(feature_list,1)

        if self.drop_rate > 0.:
            con_feature = F.dropout(con_feature, p=self.drop_rate)

        # B x F  -> B x H0
        activation = self.fc0(con_feature)
        activation = self.bn0(activation.unsqueeze(1)).squeeze(1)
        activation = F.relu(activation)

        # B x H0 -> B x H1
        gates = self.fc1(activation)
        gates = self.bn1(gates.unsqueeze(1)).squeeze(1)
        gates = F.relu(gates)
        # B x H1 -> B x H2
        gates = self.fc2(gates)
        gates = self.bn2(gates.unsqueeze(1)).squeeze(1)
        gates = torch.sigmoid(gates)

        # B x H0 -> B x H0
        activation = torch.mul(activation, gates)

        return activation

class SigmodModel(nn.Module):
    def __init__(self,hidden_size,num_classes) -> None:
        super(SigmodModel,self).__init__()
        self.logistic = nn.Linear(hidden_size, num_classes)

    def forward(self,logits):
        logits = self.logistic(logits)
        output = torch.sigmoid(logits)
        return logits,output

class LSTM_att(nn.Module):
    '''

    '''
    def __init__(self, config,input_hidden_size):
        super(LSTM_att, self).__init__()
        hidden_size = 512
        num_layers = 2
        self.lstm = nn.LSTM(input_hidden_size, hidden_size, num_layers, bidirectional=True,
          batch_first=True,
          dropout=(config.drop_rate))
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.rand(hidden_size * 2))
        # self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        emb = x
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax((torch.matmul(M, self.w)), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        return out


