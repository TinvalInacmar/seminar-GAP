from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models.classifiers.base import BaseClassifier


@CLASSIFIERS.register_module()
class MultiHeadFERClassifier(BaseClassifier):

    def __init__(self, extractor, convert, head=None, pretrained=None):
        super().__init__()
        
        self.extractor = build_backbone(extractor)
                        
        self.convert = build_neck(convert)

        self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)

    def extract_feat(self, img):
        x = self.extractor(img)
        x = self.convert(x)
        return x
    
    def forward_train(self, img, gt_label, coarse_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = self.head.forward_train(x, gt_label, coarse_label)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
    
    def inference(self, img, **kwargs):
        x = self.extract_feat(img)
        x = self.head.extract_feat(x)
        return x
    
    def aug_test(self, imgs, **kwargs): # TODO: pull request: add aug test to mmcls
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred
