from typing import Tuple, Union

import torch
from torch import Tensor

from mmdet.structures import OptSampleList, SampleList
from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector


@MODELS.register_module()
class YOLOWorldMambaKATDetector(YOLOWorldDetector):
    """YOLO-World with a trainable adapter before the frozen neck."""

    def __init__(self,
                 *args,
                 degradation_adapter: dict,
                 alignment_loss: dict = None,
                 freeze_yolo_world: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.degradation_adapter = MODELS.build(degradation_adapter)
        self.alignment_loss = (
            MODELS.build(alignment_loss)
            if alignment_loss is not None else None)
        self.freeze_yolo_world = freeze_yolo_world
        if self.freeze_yolo_world:
            self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for param in self.degradation_adapter.parameters():
            param.requires_grad = True
        self.backbone.eval()
        if self.with_neck:
            self.neck.eval()
        self.bbox_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_yolo_world:
            self.freeze_base_model()
            self.degradation_adapter.train(mode)
        return self

    @staticmethod
    def _split_paired_inputs(batch_inputs):
        if not isinstance(batch_inputs, dict):
            return None, batch_inputs

        def first_existing(keys):
            for key in keys:
                if key in batch_inputs:
                    return batch_inputs[key]
            return None

        clean_inputs = first_existing(
            ('clean_inputs', 'clean_img', 'clean'))
        degraded_inputs = first_existing(
            ('degraded_inputs', 'degraded_img', 'degraded', 'inputs'))
        if degraded_inputs is None:
            raise KeyError('Paired input dict must contain degraded_inputs, '
                           'degraded_img, degraded, or inputs.')
        return clean_inputs, degraded_inputs

    def _extract_frozen_backbone_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
        with torch.no_grad():
            txt_feats = None
            txt_masks = None
            if batch_data_samples is None:
                texts = self.texts
                txt_feats = self.text_feats
                txt_masks = getattr(self, 'text_masks', None)
            elif (isinstance(batch_data_samples, dict)
                  and 'texts' in batch_data_samples):
                texts = batch_data_samples['texts']
            elif (isinstance(batch_data_samples, list)
                  and hasattr(batch_data_samples[0], 'texts')):
                texts = [
                    data_sample.texts for data_sample in batch_data_samples
                ]
            elif hasattr(self, 'text_feats'):
                texts = self.texts
                txt_feats = self.text_feats
                txt_masks = getattr(self, 'text_masks', None)
            else:
                raise TypeError(
                    'batch_data_samples should be dict or list.')

            if txt_feats is not None:
                img_feats = self.backbone.forward_image(batch_inputs)
            else:
                img_feats, (txt_feats, txt_masks) = self.backbone(
                    batch_inputs, texts)
        return img_feats, txt_feats, txt_masks

    def _forward_frozen_neck(
            self, img_feats: Tuple[Tensor, ...],
            txt_feats: Tensor) -> Tuple[Tensor, ...]:
        if not self.with_neck:
            return img_feats
        if self.mm_neck:
            return self.neck(img_feats, txt_feats)
        return self.neck(img_feats)

    def loss(self, batch_inputs,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate detection and clean-degraded feature alignment losses."""
        self.bbox_head.num_classes = self.num_train_classes
        clean_inputs, degraded_inputs = self._split_paired_inputs(batch_inputs)

        degraded_feats, txt_feats, txt_masks = (
            self._extract_frozen_backbone_feat(degraded_inputs,
                                               batch_data_samples))
        enhanced_feats = self.degradation_adapter(degraded_feats)
        detection_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)

        bbox_head_training = self.bbox_head.training
        self.bbox_head.train()
        losses = self.bbox_head.loss(detection_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        if self.freeze_yolo_world and not bbox_head_training:
            self.bbox_head.eval()
        if clean_inputs is not None and self.alignment_loss is not None:
            clean_feats, _, _ = self._extract_frozen_backbone_feat(
                clean_inputs, batch_data_samples)
            losses['loss_cwd'] = self.alignment_loss(enhanced_feats,
                                                     clean_feats)
        return losses

    def predict(self,
                batch_inputs,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict with degraded inputs only; clean inputs are ignored."""
        _, degraded_inputs = self._split_paired_inputs(batch_inputs)
        backbone_feats, txt_feats, txt_masks = (
            self._extract_frozen_backbone_feat(degraded_inputs,
                                               batch_data_samples))
        enhanced_feats = self.degradation_adapter(backbone_feats)
        img_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)

        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs,
            batch_data_samples: OptSampleList = None):
        _, degraded_inputs = self._split_paired_inputs(batch_inputs)
        backbone_feats, txt_feats, txt_masks = (
            self._extract_frozen_backbone_feat(degraded_inputs,
                                               batch_data_samples))
        enhanced_feats = self.degradation_adapter(backbone_feats)
        img_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)
        return self.bbox_head.forward(img_feats, txt_feats, txt_masks)
