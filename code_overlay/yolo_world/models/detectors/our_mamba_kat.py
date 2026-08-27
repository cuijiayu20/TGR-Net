from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.structures import OptSampleList, SampleList
from mmyolo.registry import MODELS

from .Our import OurDetector


@MODELS.register_module()
class OurMambaKATDetector(OurDetector):
    """OW-OVD detector with a trainable Mamba-KAT degradation adapter."""

    def __init__(self,
                 *args,
                 degradation_adapter: dict,
                 alignment_loss: dict = None,
                 det_loss_weights: dict = None,
                 alignment_pairing: str = 'paired',
                 freeze_base_detector: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if alignment_pairing not in ('paired', 'shuffled'):
            raise ValueError(
                'alignment_pairing must be "paired" or "shuffled", got '
                f'{alignment_pairing!r}.')
        self.degradation_adapter = MODELS.build(degradation_adapter)
        self.alignment_loss = (
            MODELS.build(alignment_loss)
            if alignment_loss is not None else None)
        self.det_loss_weights = det_loss_weights or {}
        self.alignment_pairing = alignment_pairing
        self.freeze_base_detector = freeze_base_detector
        if self.freeze_base_detector:
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
        if self.freeze_base_detector:
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

        clean_inputs = first_existing(('clean_inputs', 'clean_img', 'clean'))
        degraded_inputs = first_existing(
            ('degraded_inputs', 'degraded_img', 'degraded', 'inputs'))
        if degraded_inputs is None:
            raise KeyError('Paired input dict must contain degraded_inputs, '
                           'degraded_img, degraded, or inputs.')
        return clean_inputs, degraded_inputs

    def _extract_frozen_backbone_feat(
            self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        with torch.no_grad():
            img_feats, _ = self.backbone(batch_inputs, None)
        return tuple(img_feats)

    def _build_frozen_text_feat(
            self, img_feats: Tuple[Tensor, ...]) -> Tensor:
        if self.reparameterized:
            return None

        with torch.no_grad():
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = F.normalize(txt_feats, dim=-1, p=2)

            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
            if self.fusion_att:
                att_feats = self.bbox_head.att_embeddings
                att_feats = att_feats[None].repeat(
                    img_feats[0].shape[0], 1, 1)
                txt_feats = torch.cat([txt_feats, att_feats], dim=1)
        return txt_feats

    def _forward_frozen_neck(
            self, img_feats: Tuple[Tensor, ...],
            txt_feats: Tensor) -> Tuple[Tensor, ...]:
        if not self.with_neck:
            return img_feats
        if self.mm_neck:
            return self.neck(img_feats, txt_feats)
        return self.neck(img_feats)

    @staticmethod
    def _scale_loss_value(value, weight: float):
        if isinstance(value, (list, tuple)):
            return [loss_item * weight for loss_item in value]
        return value * weight

    @staticmethod
    def _detach_loss_value(value):
        if isinstance(value, (list, tuple)):
            detached = [loss_item.detach().float().mean()
                        for loss_item in value]
            return torch.stack(detached).mean()
        return value.detach()

    def _apply_det_loss_weights(self, losses: dict) -> dict:
        if not self.det_loss_weights:
            return losses

        weighted_losses = dict(losses)
        for loss_name, weight in self.det_loss_weights.items():
            if loss_name not in weighted_losses:
                continue
            if weight == 0:
                raw_name = loss_name.replace('loss_', '', 1) + '_raw'
                weighted_losses[raw_name] = self._detach_loss_value(
                    weighted_losses[loss_name])
            weighted_losses[loss_name] = self._scale_loss_value(
                weighted_losses[loss_name], float(weight))
        return weighted_losses

    def loss(self, batch_inputs,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_training_classes
        clean_inputs, degraded_inputs = self._split_paired_inputs(batch_inputs)

        degraded_feats = self._extract_frozen_backbone_feat(degraded_inputs)
        txt_feats = self._build_frozen_text_feat(degraded_feats)
        enhanced_feats = self.degradation_adapter(degraded_feats)
        detection_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)

        bbox_head_training = self.bbox_head.training
        self.bbox_head.train()
        losses = self.bbox_head.loss(detection_feats,
                                     txt_feats,
                                     batch_data_samples,
                                     fusion_att=self.fusion_att)
        if self.freeze_base_detector and not bbox_head_training:
            self.bbox_head.eval()
        losses = self._apply_det_loss_weights(losses)
        if clean_inputs is not None and self.alignment_loss is not None:
            if self.alignment_pairing == 'shuffled':
                clean_inputs = clean_inputs.roll(shifts=1, dims=0)
            clean_feats = self._extract_frozen_backbone_feat(clean_inputs)
            losses['loss_cwd'] = self.alignment_loss(enhanced_feats,
                                                     clean_feats)
        return losses

    def predict(self,
                batch_inputs,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        _, degraded_inputs = self._split_paired_inputs(batch_inputs)
        backbone_feats = self._extract_frozen_backbone_feat(degraded_inputs)
        txt_feats = self._build_frozen_text_feat(backbone_feats)
        enhanced_feats = self.degradation_adapter(backbone_feats)
        img_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)

        self.bbox_head.num_classes = self.num_test_classes
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale,
                                              fusion_att=self.fusion_att)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs,
            batch_data_samples: OptSampleList = None):
        _, degraded_inputs = self._split_paired_inputs(batch_inputs)
        backbone_feats = self._extract_frozen_backbone_feat(degraded_inputs)
        txt_feats = self._build_frozen_text_feat(backbone_feats)
        enhanced_feats = self.degradation_adapter(backbone_feats)
        img_feats = self._forward_frozen_neck(enhanced_feats, txt_feats)
        return self.bbox_head.forward(img_feats, txt_feats)
