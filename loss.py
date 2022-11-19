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

        self.sizes = sizes # config.sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        # #MJ: if self.loc_loss_type == 'iou':
        #     loss = -torch.log(ious)

        # elif self.loc_loss_type == 'giou':
        self.center_loss = nn.BCEWithLogitsLoss()  #https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
        

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

# labels_batch, box_targets_batch = self.prepare_targets(locations_batch, targets_batch)  
    def prepare_targets(self, points, targets_batch): # called from labels, box_targets = self.prepare_target(locations, targets)
        ex_size_of_interest = []

        for i, points_per_level in enumerate(points): 
             # points = locations = a  feature map locations for the images in the current batch
            # sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
            size_of_interest_per_level = points_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(points_per_level), -1)  #MJ: shape =(N_{i},S)
                # MJ: size_of_interest_per_level : shape = (2,) = torch.Size( [2])
                # size_of_interest_per_level[None]: shape = (1,2)
                # size_of_interest_per_level[None].expand(len(points_per_level), -1): shape = (N_{i}, 2)
            )
        #ex_size_of_interest: a list of shape = (N_{i}, 2)
        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)

         # ex_size_of_interest: shape = (N,2) 
        n_points_per_level = [len(points_per_level) for points_per_level in points]
        point_all = torch.cat(points, dim=0)

        #point_all: shape = (N,2), N = n_points_per_level 

        labels_for_positive_anchors_batch, box_targets_for_positive_anchors_batch = self.compute_targets_for_locations(
            point_all, targets_batch, ex_size_of_interest, n_points_per_level
        )

        # labels_for_positive_anchors_batch: a list with len B with elements of shape (N,1); box_targets_for_positive_anchors_batch: a list of B elements,
        # whose shape is (N,4)
              
        for i in range(len(labels_for_positive_anchors_batch)): #MJ: labels_positive_anchors_batch:  i ranges over B
            # n_points_per_level =[ N_{1}, N_{2}, N_{3}, N_{4}, N_{5}], whose sum is N
            #torch.split: https://mopipe.tistory.com/147
            labels_for_positive_anchors_batch[i] = torch.split(labels_for_positive_anchors_batch[i], n_points_per_level, 0)
            box_targets_for_positive_anchors_batch[i] = torch.split(box_targets_for_positive_anchors_batch[i], n_points_per_level, 0)

            # labels_positive_anchors_batch[i] is a list of 5 elements
        # label_level_first = []
        # box_target_level_first = []

        
        labels_batch_levels  = []
        box_targets_batch_levels = []

        #rename:
        labels_for_positive_anchors_levels_batch = labels_for_positive_anchors_batch
        box_targets_for_positive_anchors_levels_batch = box_targets_for_positive_anchors_batch
        #Concatenate list labels_positive_anchors_batch along the level dim
        for level in range(len(points)):  # for each level on the  list of  feature maps
            labels_batch_levels.append(
                torch.cat( [labels_levels_per_img[level]  for labels_levels_per_img in labels_for_positive_anchors_levels_batch],  0)
            )
            box_targets_batch_levels.append(
                torch.cat(
                    [box_targets_per_img[level] for box_targets_per_img in box_targets_for_positive_anchors_levels_batch], 0
                )
            )

       #
        return     labels_batch_levels,   box_targets_batch_levels 
        # box_targets_batch_levels: input = [target_img1_level1, target_img2_level1,..  ]
        # The result of concatenation along dim =0:  targets_imgs_level1: shape = (sum(M_{i}), 4)
        #  targets_imgs_level2: shape = (sum(M_{i}), 4), i = ith image;
        # target_img1_level1: shape =(M_1,4),...
        # return: box_targets_batch_levels = a list of 5 elements, where an element =   (sum(M_{i}), 4)

    #    called from   labels, box_targets = self.compute_targets_for_locations(
        #     point_all, targets, ex_size_of_interest, n_points_per_level
        # )
    def compute_targets_for_locations(
        self, locations, targets_batch, ex_sizes_of_interest, n_point_per_level
    ):
        labels_for_positive_anchors_batch = []
        box_targets_for_positive_anchors_batch = []

        xs, ys = locations[:, 0], locations[:, 1] #here locations is  feature map locations: shape = (N,2)
        

        for i in range(len(targets_batch)):  # Here targets ( shape = (B, M, 4) ) is a set of M bbox targets for the images in the current batch
            targets_per_img = targets_batch[i]  #target_per_image: shape = (M,5), 4 = location, 1= label

            assert targets_per_img.mode == 'xyxy'

            bboxes = targets_per_img.box  #bboxes: shape = (M,4)

        # cf:  # dataset.py:
        # __get_items__()
        # boxes = [o['bbox'] for o in annot]
        # boxes = torch.as_tensor(boxes).reshape(-1, 4)
        # target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        # target.fields['labels'] = classes: boxes <=> classes

            labels_per_img = targets_per_img.fields['labels']  #   labels_per_img: shape =(M,C)
            area = targets_per_img.area()                      #   area: shape = (M,)

            # Similar to NumPy you can insert a singleton dimension ("unsqueeze" a dimension) by indexing this dimension with None.
            # In turn n[:, None] will have the effect of inserting a new dimension on dim=1. This is equivalent to n.unsqueeze(dim=1):

            # bboxes[:, :] represents bboxes in image i
            l = xs[:, None] - bboxes[:, 0][None]  #Shape of l = (N,M):  xs[:, None],- bboxes[:, 0][None]:
                                                  # xs[:, None]: shape=(N,1),  bboxes[:, 0]: shape=(M,) ; boxes[:,0][None]: shape=(1,M)
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_for_anchors_per_img = torch.stack([l, t, r, b], 2)  # shape of  box_targets_for_anchors_per_img = (N,M,4)

            if self.center_sample: # If the trick of sampling center points is adopted: True by default
                is_in_boxes_for_anchors = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

               # self.get_sample_region  return is_in_boxes, a boolean array

            else:
                #  min_l_lrtb_elements, args_min = box_targets_per_img.min(2); min_lrtb_elements: shape = (N,M). args_min : shape =(N,M)
                #min_lrtb_elements may contain negative values, in which case the anchor points are outside of the gt bboxes
                # In order for the anchor points to be positive, they should have positive min-lrtb:
                is_in_boxes_for_anchors = box_targets_for_anchors_per_img.min(2)[0] > 0  #MJ:  box_targets_per_img = torch.stack([l, t, r, b], 2):  (N,M,4)

            max_box_targets_for_anchors_per_img = box_targets_for_anchors_per_img.max(2)[0] #MJ: max_box_targets_per_img: shape = (N,M)
            # ex_size_of_interest: shape = (N,2) 
            is_cared_for_anchors_in_level = (
                max_box_targets_for_anchors_per_img >= ex_sizes_of_interest[:, [0]]  # max_box_targets_per_img : shape=(N,M)
            ) & (max_box_targets_for_anchors_per_img <= ex_sizes_of_interest[:, [1]]) # ex_sizes_of_interest: shape = (N,1)
            #  is_cared_in_level : a boolean matrix of shape (N,M)

            # ex_sizes_of_interest[:, [0]] = [ ex_sizes_of_interest[:, 0] ] : shape - (N,1)
            # ex_sizes_of_interest[:, [1]] = [ ex_sizes_of_interest[:, 1] ]

            gt_areas_for_anchors = area[None].repeat(len(locations), 1)  # area[None]: shape =(1,M)
            #   gt_areas_for_anchors : shape = (N,M)

            gt_areas_for_anchors[is_in_boxes_for_anchors == 0] = INF  # is_in_boxes: a boolean matrix of shape (N,M)

            gt_areas_for_anchors[is_cared_for_anchors_in_level == 0] = INF  #is_cared_in_level: a boolean matrix of shape (N,M)

            min_areas_for_anchors, indices_to_min_areas_for_anchors = gt_areas_for_anchors.min(1)
            #_min_areas_for_anchors: shape = (N,), indices_to_min_areas: shape = (N,), its values range over M

            min_box_targets_for_anchors_per_img = box_targets_for_anchors_per_img[
                range(len(locations)),  indices_to_min_areas_for_anchors   # box_targets_for_anchors_per_img = (N,M,4);  
            ]
            # min_box_targets_for_anchors_per_img: shape = (N,4)
            # labels_per_img: shape = (M,1), The dim 1 ranges over the class indices.
            min_labels_for_anchors_per_img = labels_per_img[ indices_to_min_areas_for_anchors ]
            # min_labels_for_anchors_per_img : shape = (N,1)
            min_labels_for_anchors_per_img[min_areas_for_anchors == INF] = 0 # min_labels_per_img may have INF element

            labels_for_positive_anchors_batch.append(min_labels_for_anchors_per_img) # min_labels_per_img:shape =(N,1)
            box_targets_for_positive_anchors_batch.append(min_box_targets_for_anchors_per_img)
        #END  for i in range(len(targets_batch))

        #
        # labels_for_positive_anchors_batch: a list with len B with elements of shape (N,1); box_targets_for_positive_anchors_batch: a list of B elements,
        # whose shape is (N,4)
        return labels_for_positive_anchors_batch,  box_targets_for_positive_anchors_batch

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
    def forward(self, locations, cls_pred_batch, box_pred_batch, center_pred_batch, targets_batch):
        batch_size = cls_pred_batch[0].shape[0]
        n_class = cls_pred_batch[0].shape[1]

        #labels_levels_batch, box_targets_levels_batch: locations is a list of anchor points on each feature map.
        
        labels_batch_levels, box_targets_batch_levels = self.prepare_targets(locations, targets_batch)  # locations is  feature map locations for the images in the the batch

        # box_targets_batch_levels = a list of 5 elements, where an element =   (sum(M_{i}), 4)
       
        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []

        #cls_pred_batch[i] = the output the classification head for feature map i
        for i in range(len(  labels_batch_levels)):  # i ranges over the feature map levels, ie. range( 5) 
            cls_flat.append(cls_pred_batch[i].permute(0, 2, 3, 1).reshape(-1, n_class))  #(B,C,H,W) => (B,H,W,C)
            box_flat.append(box_pred_batch[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred_batch[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels_batch_levels[i].reshape(-1))
            box_targets_flat.append(box_targets_batch_levels[i].reshape(-1, 4))

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
