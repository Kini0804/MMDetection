# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):

    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')
    # CLASSES = ('ore carrier', 'bulk cargo carrier', 'general cargo ship', 'container ship', 'fishing boat',
    #             'passenger ship', 'ship', 'aircraft carrier', 'warcraft', 'merchant ship',
    #             'Nimitz class aircraft carrier', 'Enterprise class aircraft carrier', 'Arleigh Burke class destroyers', 'WhidbeyIsland class landing craft', 'Perry class frigate',
    #            'Sanantonio class amphibious transport dock', 'Ticonderoga class cruiser', 'Kitty Hawk class aircraft carrier', 'Admiral Kuznetsov aircraft carrier', 'Abukuma-class destroyer escort',
    #            'Austen class amphibious transport dock', 'Tarawa-class amphibious assault ship', 'USS Blue Ridge (LCC-19)', 'OXo|--)', 'Car carrier', 'submarine',
    #            'lute', 'Medical ship', 'Ford-class aircraft carriers', 'Midway-class aircraft carrier', 'Invincible-class aircraft carrier')
    # CLASSES = ('ship', 'aircraft carrier', 'warcraft', 'merchant ship', 'Nimitz class aircraft carrier', 'Enterprise class aircraft carrier', 'Arleigh Burke class destroyers',
    # 'WhidbeyIsland class landing craft', 'Perry class frigate', 'Sanantonio class amphibious transport dock', 'Ticonderoga class cruiser',
    # 'Kitty Hawk class aircraft carrier', 'Admiral Kuznetsov aircraft carrier', 'Abukuma-class destroyer escort', 'Austen class amphibious transport dock',
    # 'Tarawa-class amphibious assault ship', 'USS Blue Ridge (LCC-19)', 'Container ship', 'OXo|--)', 'Car carrier([]==[])', 'Hovercraft', 'yacht', 'Container ship(_|.--.--|_]=',
    # 'Cruise ship', 'submarine', 'lute', 'Medical ship', 'Car carrier(======|', 'Ford-class aircraft carriers', 'Midway-class aircraft carrier', 'Invincible-class aircraft carrier')
    CLASSES = ('船', '航母', '军舰', '商船', '尼米兹级航母', '企业级航母', '阿利伯克级驱逐舰', '惠德贝岛级船坞登陆舰', '佩里级护卫舰', '圣安东尼奥级两栖船坞运输舰', '提康德罗加级巡洋舰', '小鹰级航母', '俄罗斯库兹涅佐夫号航母', '阿武隈级护卫舰', '奥斯汀级两栖船坞运输舰', '塔拉瓦级通用两栖攻击舰', '蓝岭级指挥舰', '集装箱货船', '尾部OX头部圆指挥舰', '运输汽车船([]==[])', '气垫船', '游艇', '货船(_|.--.--|_]=', '游轮', '潜艇', '琵琶形军舰', '医疗船', '运输汽车船(======|', '福特级航空母舰', '中途号航母', '无敌级航空母舰')
    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
