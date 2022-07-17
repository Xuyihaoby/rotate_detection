from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import mmcv
from mmdet.core.visualization import imshow_det_rbboxes,imshow_det_bboxes
import warnings
import numpy as np


@DETECTORS.register_module()
class RSparseRCNN(TwoStageDetector):
    r"""Implementation of `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self, with_queryrpn=False, *args, **kwargs):
        self.with_queryrpn=with_queryrpn
        super(RSparseRCNN, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'Sparse R-CNN do not support external proposals'

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      hor_gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'Sparse R-CNN does not instance segmentation'

        x = self.extract_feat(img)
        if not self.with_queryrpn:
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(x, img_metas)
            roi_losses = self.roi_head.forward_train(
                x,
                proposal_boxes,
                proposal_features,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                imgs_whwh=imgs_whwh)
            return roi_losses
        elif self.with_queryrpn:
            losses = dict()
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_boxes, proposal_features, imgs_whwh = self.rpn_head.forward_train(
                x,
                img_metas,
                hor_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            roi_losses = self.roi_head.forward_train(
                x,
                proposal_boxes,
                proposal_features,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                imgs_whwh=imgs_whwh)
            losses.update(roi_losses)
            return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bbox_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    version='v1'):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes

        if bboxes.shape[1] == 6:
            imshow_det_rbboxes(
                img,
                bboxes,
                labels,
                class_names=self.CLASSES,
                score_thr=score_thr,
                bbox_color=bbox_color,
                text_color=text_color,
                thickness=thickness,
                font_scale=font_scale,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                version=version)
        elif bboxes.shape[1] == 5:
            imshow_det_bboxes(
                img,
                bboxes,
                labels,
                # segms,
                class_names=self.CLASSES,
                score_thr=score_thr,
                bbox_color="white",
                text_color="black",
                #mask_color=mask_color,
                thickness=thickness,
                font_scale=font_scale,
                #font_size=font_size,
                win_name=win_name,
                #fig_size=fig_size,
                show=show,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
