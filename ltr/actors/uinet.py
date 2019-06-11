from . import BaseActor
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
import matplotlib.pyplot as plt
import torch
from datetime import datetime

class UIActor(BaseActor):
    """ Actor for training the ui-Net"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'traing_mask','gt'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        train_img=data['train_images']
        print("train_img:",train_img[0].shape)
        img=self.net(data['train_images'])
         
        print("img:",img.shape) #240*320
        #print(img)
        #plt.imshow(img)
        #plt.show()
        SR_probs = F.sigmoid(img)
        print("SR_probs.shape",SR_probs.shape)

        #loader = F.Compose([transforms.ToTensor()])  
        #unloader = tvF.to_pil_image()
        
        indices =torch.tensor([0])
        
        SR_flat = SR_probs.view(SR_probs.shape,-1)
        SR_flat=SR_flat.to(self.device)
        #print("SR_flat",SR_flat)
        #plt.imshow(GT)
        #plt.show()
        GT=data['train_mask']
        
        print("GT.shape",GT.shape)
        #GT.
        GT_flat = GT.view(GT.shape,-1)

        dd=datetime.now().strftime("%Y%m%d_%H%M%S")
        sec=datetime.now().strftime("%S")
        if(int(sec)%59==0):
            bb= data['train_anno']
   
            plt.subplot(1,3,1)
            tra=train_img.clone()
            pic=torch.index_select(tra.cpu(),1,indices)
            pic=pic.view(pic.shape[2],pic.shape[3],pic.shape[4])
            pic=tvF.to_pil_image(pic)
            #plt.Rectangle(self, xy, width, height)
            plt.imshow(pic)
            #plt.show()
            #plt.savefig("/home/peter/PycharmProjects/pytracking/ltr/or_pic1"+dd+".jpg")
            
            plt.subplot(1,3,2)
            ak=SR_probs.clone()
            pic=torch.index_select(ak.cpu(),1,indices)
            pic=pic.view(pic.shape[1],pic.shape[2],pic.shape[3])
            pic=tvF.to_pil_image(pic)
            plt.imshow(pic)

            #plt.show()
            #plt.savefig("/home/peter/PycharmProjects/pytracking/ltr/or_pic2"+dd+".jpg")
            plt.subplot(1,3,3)
            ag=GT_flat.clone()
            pic=torch.index_select(ag.cpu(),1,indices)
            pic=pic.view(pic.shape[1],pic.shape[2],pic.shape[3])
            pic=tvF.to_pil_image(pic)
            plt.imshow(pic)
            #plt.show()
            plt.savefig("/home/peter/PycharmProjects/pytracking/ltr/jpg/GT_pic"+dd+".jpg")

        GT_flat=GT_flat.to(self.device)
        #print(GT)
        loss =self.objective(SR_flat,GT_flat)
        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}
        """
        mask_pred = self.net(data['train_images'])
        
        mask_pred = mask_pred.view(-1, mask_pred.shape[1])
        mask_gt = data['train_mask'].view(-1, data['train_mask'].shape[1])

        # Compute loss
        loss = self.objective(mask_pred, mask_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}
        """
        return loss, stats