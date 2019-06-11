import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
import ltr.models.masknet.unet as UNET

class UInet(nn.Module):
    """ UI network module"""
    def __init__(self, feature_extractor, unet, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            unet - mask prediction module
            encoder_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(UInet, self).__init__()

        self.feature_extractor = feature_extractor
        self.unet = unet
        self.bb_regressor_layer = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)


    # def forward(self, train_imgs):
    #     """ Forward pass
    #     Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
    #     corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
    #     #输出
    #     """
    #     num_sequences = train_imgs.shape[-4]
    #     num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
    #     #num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

    #     # Extract backbone features
    #     train_feat = self.extract_backbone_features(
    #         train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
    #     #test_feat = self.extract_backbone_features(
    #     #    test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

    #     # For clarity, send the features to bb_regressor in sequence form, i.e. [sequence, batch, feature, row, col]
    #     train_feat_seq = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
    #                       for feat in train_feat.values()]
    #     #test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
    #                      #for feat in test_feat.values()]

    #     # Obtain iou preAtomIoU
    #     mask_pred = self.unet(train_feat_seq)
    #     return mask_pred


    def forward(self, train_imgs):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        #输出
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        
        train_imgs=train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1])
        train_feat = self.extract_backbone_features(train_imgs)

        # Obtain iou preAtomIoU
        print("len(train_feat)",len(train_feat))
        #print("train_feat ",train_feat )
        train_feat_s = [feat.view(num_train_images*num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in train_feat.values()]
        print("train_feat_s",train_feat_s[0].shape)
        mask_pred = self.unet(train_feat_s)
        return mask_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        #print(layers)
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)




@model_constructor
def uinet_resnet18(backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained,output_layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4'])

    # Bounding box regressor
    mask_predictor = UNET.U_Net()

    net = UInet(feature_extractor=backbone_net, unet=mask_predictor,
                  extractor_grad=False)
    
    return net



