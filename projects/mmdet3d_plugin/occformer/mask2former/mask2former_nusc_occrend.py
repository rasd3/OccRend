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
class Mask2FormerNuscOccRendHead(Mask2FormerNuscOccHead):
    """Implements the Mask2FormerOccHead + PointRend
    """

    def __init__(self,
                 num_rend_points=8096,
                 loss_rend=None,
                 sampling_method='uncertainty',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rend_mlp = nn.Conv1d(209, self.num_classes, 1) # num of channels + 192
        self.num_rend_points = num_rend_points
        self.loss_rend = build_loss(loss_rend)
        self.sampling_method = sampling_method
        
    def forward_occrend(self, coarse, fine, gt_occ, points, img_metas, coarse2):
        pc_range = torch.tensor(self.point_cloud_range).cuda()
        gt_occ.masked_fill_(gt_occ == 255, 0)
        if self.sampling_method == 'uncertainty':
            # for uncertainty sampling 
            with torch.no_grad():
                occ_point_coords = get_nusc_lidarseg_point_coords(coarse, 
                    points, [torch.tensor([0]).cuda()], self.num_rend_points, 
                    self.oversample_ratio, self.importance_sample_ratio, self.point_cloud_range, 
                    padding_mode=self.padding_mode, rend=True)
        elif self.sampling_method == 'gt':
            b_size = coarse2.shape[0]
            occ_size = torch.tensor(coarse2.shape[2:]).cuda()
            voxel_size = [(pc_range[i+3] - pc_range[i])/occ_size[i] for i in range(3)]
            voxel_size = torch.tensor(voxel_size).cuda()
            occ_point_coords = []
            for b in range(b_size):
                gt_occ_oh = F.one_hot(gt_occ[b], num_classes=17).to(torch.float32).permute(3, 0, 1, 2)
                output_voxel = coarse2[b]
                disc = (gt_occ_oh - output_voxel).abs().sum(0)
                dims = disc.shape

                flat_disc = torch.flatten(disc)
                d_val, d_idx = torch.topk(flat_disc, self.num_rend_points)

                indices = [torch.div(d_idx, (dims[1]*dims[2]), rounding_mode='floor'),
                           torch.div(d_idx % (dims[1]*dims[2]), dims[2], rounding_mode='floor'),
                           d_idx % dims[2]]
                indices = torch.vstack(indices).permute(1, 0)
                unc_coor = indices * voxel_size + pc_range[:3]
                occ_point_coords.append(((unc_coor - pc_range[:3]) / (pc_range[3:] - pc_range[:3])).unsqueeze(0))
            occ_point_coords = torch.cat(occ_point_coords, dim=0)
        else:
            raise NotImplementedError('Not implemented sampling method')

        if False:
            import cv2
            from mmdet3d.core.visualizer.image_vis import project_pts_on_img
            vis_points = occ_point_coords.squeeze().clone()
            vis_points = vis_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
            for idx, img_meta in enumerate(img_metas):
                lidar2img = img_meta['lidar2img']
                img_filenames = img_meta['img_filenames']
                for idx, cam_type in enumerate(list(img_filenames.keys())):
                    print(cam_type)
                    img = cv2.imread(img_filenames[cam_type])
                    proj_img = project_pts_on_img(vis_points.detach().cpu().numpy(),
                                                  img, lidar2img[idx], with_gui=False)
                    cv2.imwrite('test_%02d.png' % idx, proj_img)

        coarse_p = point_sample_3d(coarse, occ_point_coords, padding_mode=self.padding_mode)
        fine_p = point_sample_3d(fine, occ_point_coords, padding_mode=self.padding_mode)
        feats_p = torch.cat([coarse_p, fine_p], dim=1)
        rends_p = self.rend_mlp(feats_p)
        gt_occ_p = point_sample_3d(gt_occ.unsqueeze(1).to(torch.float), 
                                   occ_point_coords, padding_mode=self.padding_mode,
                                   mode='nearest').squeeze(1).to(torch.long)
        if type(self.loss_rend).__name__ == 'DiceLoss':
            gt_occ_p = F.one_hot(gt_occ_p, num_classes=self.num_classes).permute(0, 2, 1)
        loss_occ = self.loss_rend(rends_p, gt_occ_p)

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

        mask_preds = F.interpolate(
            all_mask_preds[-1],
            size=tuple(img_metas[0]['occ_size']),
            mode='trilinear',
            align_corners=self.align_corners,
        )
        coarse2 = self.format_results(all_cls_scores[-1], mask_preds)

        losses_occrend = self.forward_occrend(coarse, fine, gt_occ, points, img_metas, coarse2)
        losses.update(losses_occrend)

        # dbg
        if False:
            import cv2
            from mmdet3d.core.visualizer.image_vis import project_pts_on_img

            b_size = all_mask_preds[-1].shape[0]
            mask_preds = F.interpolate(
                all_mask_preds[-1],
                size=tuple(img_metas[0]['occ_size']),
                mode='trilinear',
                align_corners=self.align_corners,
            )
            output_voxels = self.format_results(all_cls_scores[-1], mask_preds)
            gt_occ_ohs = F.one_hot(gt_occ, num_classes=17).to(torch.float32).permute(0, 4, 1, 2, 3)
            pc_range = torch.tensor(self.point_cloud_range).cuda()
            occ_size = torch.tensor(output_voxels.shape[2:]).cuda()
            voxel_size = [(pc_range[i+3] - pc_range[i])/occ_size[i] for i in range(3)]
            voxel_size = torch.tensor(voxel_size).cuda()

            for b in range(b_size):
                output_voxel, gt_occ_oh = output_voxels[b], gt_occ_ohs[b]
                N_P = self.num_rend_points

                # uncertatiny from gt - pred
                disc = (gt_occ_oh - output_voxel).abs().sum(0)
                dims = disc.shape
                flat_disc = torch.flatten(disc)
                d_val, d_idx = torch.topk(flat_disc, N_P)

                indices = [torch.div(d_idx, (dims[1]*dims[2]), rounding_mode='floor'),
                           torch.div(d_idx % (dims[1]*dims[2]), dims[2], rounding_mode='floor'),
                           d_idx % dims[2]]
                indices = torch.vstack(indices).permute(1, 0)
                unc_coor = indices * voxel_size + pc_range[:3]

                # uncertainty from pred 
                gt_class_logits = output_voxel.clone()
                gt_class_logits, _ = gt_class_logits.sort(0, descending=True)
                uncertainty_map = -1 * (gt_class_logits[0] - gt_class_logits[1])
                flat_unsc = torch.flatten(uncertainty_map)
                u_val, u_idx = torch.topk(flat_unsc, N_P)

                indices2 = [torch.div(u_idx, (dims[1]*dims[2]), rounding_mode='floor'),
                           torch.div(u_idx % (dims[1]*dims[2]), dims[2], rounding_mode='floor'),
                           u_idx % dims[2]]
                indices2 = torch.vstack(indices2).permute(1, 0)
                unc_coor2 = indices2 * voxel_size + pc_range[:3]

                img_meta = img_metas[b]
                lidar2img = img_meta['lidar2img']
                img_filenames = img_meta['img_filenames']
                unc_coor[:, 2] *= -1
                for idx, cam_type in enumerate(list(img_filenames.keys())):
                    print(cam_type)
                    img = cv2.imread(img_filenames[cam_type])
                    proj_img = project_pts_on_img(unc_coor.detach().cpu().numpy(),
                                                  img, lidar2img[idx], with_gui=False)
                    cv2.imwrite('test_%02d_gt.png' % idx, proj_img)

                    proj_img2 = project_pts_on_img(unc_coor2.detach().cpu().numpy(),
                                                  img, lidar2img[idx], with_gui=False)
                    cv2.imwrite('test_%02d_pred.png' % idx, proj_img2)

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
        points_original = copy.deepcopy(points)
        if True:
            output_voxels = self.format_results(mask_cls_results, mask_pred_results)
            # rescale mask prediction by OccRend
            N = self.num_rend_points
            while output_voxels.shape[-1] != img_metas[0]['occ_size'][-1]:
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
        else:
            mask_pred_results_int = F.interpolate(
                mask_pred_results,
                size=tuple(img_metas[0]['occ_size']),
                mode='trilinear',
                align_corners=self.align_corners,
            )
            output_voxels_int = self.format_results(mask_cls_results, mask_pred_results_int)

        res = {
            'output_voxels': [output_voxels],
            'output_points': None,
        }
        res['output_points'] = self.forward_lidarseg(
            cls_preds=all_cls_scores[-1],
            mask_preds=all_mask_preds[-1],
            points=points_original,
            img_metas=img_metas,
            voxel_rends=output_voxels
        )
        return res
