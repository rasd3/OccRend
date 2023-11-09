# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (get_nusc_lidarseg_point_coords,
                               get_point_coords_infer,
                               preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.mmdet3d_plugin.utils import per_class_iu, fast_hist_crop
import pdb
import copy
from .mask2former_nusc_occ import Mask2FormerNuscOccHead

# Mask2former + PointRend for 3D Occupancy Segmentation on nuScenes dataset
@HEADS.register_module()
class Mask2FormerNuscOccRendHead_bw(Mask2FormerNuscOccHead):
    """Implements the Mask2FormerOccHead + PointRend
    """

    def __init__(self,
                 num_rend_points=110000,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rend_mlp = nn.Conv1d(209, self.num_classes, 1) # num of channels + 192
        self.num_rend_points = num_rend_points
        
    def forward_occrend(self, coarse, fine, gt_occ, points, img_metas):
        with torch.no_grad():
            occ_point_coords = get_nusc_lidarseg_point_coords(coarse, 
                points, [torch.tensor([0]).cuda()], self.num_points, 
                self.oversample_ratio, self.importance_sample_ratio, self.point_cloud_range, 
                padding_mode=self.padding_mode)
        
        if True:
            import cv2
            from mmdet3d.core.visualizer.image_vis import project_pts_on_img
            pc_range = torch.tensor(self.point_cloud_range).cuda()
            vis_points = occ_point_coords.squeeze().clone()
            vis_points = vis_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
            vis_points[:, 2] *= -1
            for idx, img_meta in enumerate(img_metas):
                lidar2img = img_meta['lidar2img']
                img_filenames = img_meta['img_filenames']
                for idx, cam_type in enumerate(list(img_filenames.keys())):
                    img = cv2.imread(img_filenames[cam_type])
                    proj_img = project_pts_on_img(vis_points.detach().cpu().numpy(),
                                                  img, lidar2img[idx], with_gui=False)
                    cv2.imwrite('test_%02d.png' % idx, proj_img)
            pdb.set_trace()

        coarse_p = point_sample_3d(coarse, occ_point_coords, padding_mode=self.padding_mode)
        fine_p = point_sample_3d(fine, occ_point_coords, padding_mode=self.padding_mode)
        feats_p = torch.cat([coarse_p, fine_p], dim=1)
        rends_p = self.rend_mlp(feats_p)
        gt_occ.masked_fill_(gt_occ == 255, 0)
        gt_occ_p = point_sample_3d(gt_occ.unsqueeze(1).to(torch.float), 
                                   occ_point_coords, padding_mode=self.padding_mode,
                                   mode='nearest').squeeze(1).to(torch.long)
        loss_occ = F.cross_entropy(rends_p, gt_occ_p)

        loss_dict = {}
        loss_dict['loss_occrend'] = loss_occ
        return loss_dict

    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            points=None,
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # forward
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_occ, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, points, img_metas)

        # OccRend
        coarse = self.format_results(all_cls_scores[-1], all_mask_preds[-1])
        fine = voxel_feats[0]

        losses_occrend = self.forward_occrend(coarse, fine, gt_occ, points, img_metas)
        losses.update(losses_occrend)

        # forward_lidarseg
        losses_lidarseg = self.forward_lidarseg(all_cls_scores[-1], all_mask_preds[-1], points, img_metas)
        losses.update(losses_lidarseg)

        return losses

    def simple_test(self, 
            voxel_feats,
            img_metas,
            points=None,
            **kwargs,
        ):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        points_original = copy.deepcopy(points)
        # import pdb; pdb.set_trace()
        # rescale mask prediction by OccRend
        N = 110000
        # N = points_original[0].shape[0]
        while output_voxels.shape[-1] != img_metas[0]['occ_size'][-1]:
            print(output_voxels.shape, img_metas[0]['occ_size'])
            output_voxels = F.interpolate(output_voxels, 
                                          scale_factor=2, mode='trilinear',
                                          align_corners=self.align_corners)
            
            point_idx, points = get_point_coords_infer(output_voxels, N)
            coarse_p = point_sample_3d(output_voxels, points, align_corners=False)
            fine_p = point_sample_3d(voxel_feats[0], points, align_corners=False)

            feats_p = torch.cat([coarse_p, fine_p], dim=1)
            rend = self.rend_mlp(feats_p)
            B, C, H, W, Z = output_voxels.shape
            points_idx = point_idx.unsqueeze(1).expand(-1, C, -1)
            output_voxels = (output_voxels.reshape(B, C, -1)
                             .scatter_(2, points_idx, rend)
                             .view(B, C, H, W, Z))
            # print('Here:',output_voxels.shape, img_metas[0]['occ_size'])
            # import pdb; pdb.set_trace()
        res = {
            'output_voxels': [output_voxels],
            'output_points': None,
        }
        res['output_points'] = self.forward_lidarseg(
            cls_preds=all_cls_scores[-1],
            mask_preds=all_mask_preds[-1],
            points=points_original,
            img_metas=img_metas,
        )
        return res