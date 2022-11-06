import torch
from torch import nn


INF = 100000000


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(
        self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius
    ):
        super().__init__()

        self.sizes = sizes

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        # #MJ: if self.loc_loss_type == 'iou':
        #     loss = -torch.log(ious)

        # elif self.loc_loss_type == 'giou':
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

# labels_batch, box_targets_batch = self.prepare_targets(locations_batch, targets_batch)  
    def prepare_targets(self, points_batch, targets_batch): # called from labels, box_targets = self.prepare_target(locations, targets)
        ex_size_of_interest = []

        for i, points_per_level in enumerate(points_batch): 
             # points = locations = a  feature map locations for the images in the current batch

            size_of_interest_per_level = points_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_points_per_level = [len(points_per_level) for points_per_level in points_batch]
        point_all_batch = torch.cat(points_batch, dim=0)

        labels_batch, box_targets_batch = self.compute_targets_for_locations(
            point_all_batch, targets_batch, ex_size_of_interest, n_points_per_level
        )

        for i in range(len(labels_batch)):
            labels_batch[i] = torch.split(labels_batch[i], n_points_per_level, 0)
            box_targets_batch[i] = torch.split(box_targets_batch[i], n_points_per_level, 0)

        # label_level_first = []
        # box_target_level_first = []

        labels_levels_batch  = []
        box_targets_levels_batch = []


        for level in range(len(points_batch)):  # a list of  feature map points for the images in the current batch
            labels_levels_batch.append(
                torch.cat([labels_per_img[level] for labels_per_img in labels_batch], 0)
            )
            box_targets_levels_batch.append(
                torch.cat(
                    [box_targets_per_img[level] for box_targets_per_img in box_targets_batch], 0
                )
            )

        return labels_levels_batch, box_targets_levels_batch

    
    # called from   labels, box_targets = self.compute_targets_for_locations(
        #     point_all, targets, ex_size_of_interest, n_points_per_level
        # )
    def compute_targets_for_locations(
        self, locations_batch, targets_batch, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []

        xs, ys = locations_batch[:, 0], locations_batch[:, 1] #here locations is  feature map locations for the images in the current batch
        

        for i in range(len(targets_batch)):  # Here targets is a set of bbox targets for the images in the current batch
            targets_per_img = targets_batch[i]
            assert targets_per_img.mode == 'xyxy'

            bboxes = targets_per_img.box

            labels_per_img = targets_per_img.fields['labels']
            area = targets_per_img.area()

            # Similar to NumPy you can insert a singleton dimension ("unsqueeze" a dimension) by indexing this dimension with None.
            # In turn n[:, None] will have the effect of inserting a new dimension on dim=1. This is equivalent to n.unsqueeze(dim=1):

            # bboxes[:, 0] represents bboxes in image i
            l = xs[:, None] - bboxes[:, 0][None]  #Shape of l = (N,1):  xs[:, None] - bboxes[:, 0][None] =>":" dim refers to the batch size
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)  # shape of  box_targets_per_img = (N,1,4)

            if self.center_sample: # If the trick of sampling center points is adopted: True by default
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

               # self.get_sample_region  return is_in_boxes, a boolean array

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0  #MJ:  box_targets_per_img = torch.stack([l, t, r, b], 2):  (N,1,4)

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations_batch), 1)

            locations_to_gt_area[is_in_boxes == 0] = INF

            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations_batch)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def get_sample_region(self, gt_bboxes, strides, n_point_per_level, xs, ys, radius=1): # radius = 1.5 by default
        n_gt = gt_bboxes.shape[0]
        n_loc = len(xs)
        gt = gt_bboxes[None].expand(n_loc, n_gt, 4)  # expand the singleton dim of gt_bboxes by n_loc
        
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes    

# center_targets = self.compute_centerness_targets(box_targets_flat)
    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
            top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

#  MJ: 
#       cls_pred_batch, box_pred_batch, center_pred_batch = self.head(features_batch)
#         # print(cls_pred, box_pred, center_pred)
#       locations_batch = self.compute_locations(features_batch)

#       if self.training:
#             loss_cls, loss_box, loss_center = self.loss(
#                 locations_batch, cls_pred_batch, box_pred_batch, center_pred_batch, targets_batch
#             )
         #self.loss = FCOSLoss.forward():
    def forward(self, locations_batch, cls_pred_batch, box_pred_batch, center_pred_batch, targets_batch):
        batch_size = cls_pred_batch[0].shape[0]
        n_class = cls_pred_batch[0].shape[1]


        labels_batch, box_targets_batch = self.prepare_targets(locations_batch, targets_batch)  # locations is  feature map locations for the images in the the batch


        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels_batch)):
            cls_flat.append(cls_pred_batch[i].permute(0, 2, 3, 1).reshape(-1, n_class))  #(B,C,H,W) => (B,H,W,C)
            box_flat.append(box_pred_batch[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred_batch[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels_batch[i].reshape(-1))
            box_targets_flat.append(box_targets_batch[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch_size) # tensor.numel() = the total num of elements

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss
