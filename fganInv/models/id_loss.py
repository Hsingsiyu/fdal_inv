import torch
from torch import nn
from models.model_irse import Backbone
# from model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('./models/pretrain/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        # self.cosloss = torch.nn.CosineEmbeddingLoss()
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        # cos_target = torch.ones((n_samples, 1)).float().cuda()
        loss = 0
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)

        for i in range(5):
            # print(y_feats[i].shape)
            y_feat_detached = y_feats[i].detach()
            loss +=torch.mean((y_feat_detached-y_hat_feats[i])**2)#self.cosloss(y_feat_detached, y_hat_feats[i], cos_target)

        return loss

if __name__=='__main__':
    id_loss=IDLoss().eval().cuda()
    bs=16
    a=torch.randn(bs,3,512,256).cuda()
    b=torch.randn(bs,3,512,256).cuda()
    val=id_loss(a,b)
    print(val)