from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_bone import Image_bone,Text_bone
from models.NeXtVLAD import SeFusion,SigmodModel,NeXtVLAD,LSTM_att

class FusionModel(nn.Module):

    def __init__(self,config):
        super(FusionModel, self).__init__()

        # 图像
        # self.image_model = Image_bone(config)
        # 文本
        # self.text_model = Text_bone(config)
        # 视频
        self.video_model = NeXtVLAD(dim=config.video_dim, num_clusters=config.num_clusters, 
                                    lamb=config.lamb, groups=config.groups,max_frames=config.video_max_frames)
        # 音频
        self.audio_model = NeXtVLAD(dim=config.audio_dim, num_clusters=config.num_clusters, 
                                    lamb=config.lamb, groups=config.groups,max_frames=config.audio_max_frames)

        # fusion model
        line_hidden = config.audio_hidden + config.video_hidden
        self.fusion_model = SeFusion(config,line_hidden)
        self.classifier = SigmodModel(config.hidden_size,config.num_classes)
 

    def forward(self,data,device):
        video_feature = data['video_feature'].to(device)
        audio_feature = data['audio_feature'].to(device)
        # image_feature = data['image_feature'].to(device)
        # text_feature = data['text_feature']
        # ids, mask, token_type_ids = text_feature.values()


        # image_feature,_ = self.image_model(image_feature)
        # text_feature = self.text_model(ids.to(device), mask.to(device), token_type_ids.to(device))
        video_feature = self.video_model(video_feature)
        audio_feature = self.audio_model(audio_feature)

        fusion_feature = self.fusion_model([video_feature,audio_feature])
        # fusion_feature = self.fusion_model([text_feature,video_feature,audio_feature])
        # fusion_feature = self.fusion_model([image_feature,text_feature,video_feature,audio_feature])
        logits,output = self.classifier(fusion_feature)
        # return logits,output
        return {"all_logits":logits,"all_output":output,"text_logits":0,"text_output":0}


class FusionModel_2(nn.Module):
    '''
    使用双向lstm+attention 进行视频和音频特征提取
    '''
    def __init__(self,config):
        super(FusionModel_2, self).__init__()

        # 图像
        # self.image_model = Image_bone(config)
        # 文本
        self.text_model = Text_bone(config)
        text_feature_hidden_size = 768
        # self.text_head = SigmodModel(text_feature_hidden_size,config.num_classes)
        # 视频
        self.video_model = LSTM_att(config,768)
        # 音频
        self.audio_model = LSTM_att(config,128)

        # fusion model
        self.fusion_model = SeFusion(config)
        self.classifier = SigmodModel(config.hidden_size,config.num_classes)
 

    def forward(self,data,device):
        video_feature = data['video_feature'].to(device)
        audio_feature = data['audio_feature'].to(device)
        image_feature = data['image_feature'].to(device)
        text_feature = data['text_feature']
        ids, mask, token_type_ids = text_feature.values()


        # image_feature,_ = self.image_model(image_feature)

        text_feature = self.text_model(ids.to(device), mask.to(device), token_type_ids.to(device))
        # text_logits,text_output = self.text_head(text_feature)

        video_feature = self.video_model(video_feature)
        audio_feature = self.audio_model(audio_feature)


        fusion_feature = self.fusion_model([text_feature,video_feature,audio_feature])
        logits,output = self.classifier(fusion_feature)
        return {"all_logits":logits,"all_output":output,"text_logits":0,"text_output":0}





