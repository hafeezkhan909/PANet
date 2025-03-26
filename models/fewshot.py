"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict
import shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .vgg import Encoder
from PIL import Image
import torchvision.transforms as T

def visualize_prediction(image, prediction):
    """
    Overlay the segmentation prediction on the image.
    Args:
        image (np.ndarray): Original query image (H x W x 3).
        prediction (np.ndarray): Predicted segmentation map (H x W).
    Returns:
        np.ndarray: Image with prediction overlay.
    """
    # Check if image is properly loaded
    if image is None:
        raise ValueError("Input image is None.")
    
    # Ensure the image is in RGB format (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ensure prediction is a 2D array
    if len(prediction.shape) != 2:
        raise ValueError("Prediction mask must be a 2D array.")

    # Automatically create a color map for all classes in prediction
    num_classes = prediction.max() + 1
    np.random.seed(0)  # For reproducibility
    color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Ensure background is black

    # Convert prediction mask to an RGB overlay
    overlay = color_map[prediction]

    # Resize the overlay to match the size of the input image (if necessary)
    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend the original image and the overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return blended


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))

    def save_hard_samples(self, supp_imgs, qry_imgs, pred, gt_mask, save_path, 
                      query_image_names, support_image_names, img_size):
        """
        Save hard samples to disk if hardness score exceeds the threshold.
        """
        os.makedirs(save_path, exist_ok=True)

        # Find the next available folder number
        existing_folders = [int(name) for name in os.listdir(save_path) if name.isdigit()]
        next_folder_num = max(existing_folders) + 1 if existing_folders else 1
        sample_folder = os.path.join(save_path, str(next_folder_num))
        
        # Create directories for support and query
        support_folder = os.path.join(sample_folder, 'support')
        query_folder = os.path.join(sample_folder, 'query')
        os.makedirs(support_folder, exist_ok=True)
        os.makedirs(query_folder, exist_ok=True)

        # Hardcoded directories for images and labels
        image_dir = './VOCdevkit/VOC2012/JPEGImages'
        label_dir = './VOCdevkit/VOC2012/SegmentationClassAug'

        # Save Query Images & Labels
        for idx, (qry_img, qry_name, pred_img, gt_img) in enumerate(zip(
                qry_imgs, query_image_names, pred, gt_mask)):
            
            # Extract the actual filename (removes path and extension)
            qry_filename = os.path.splitext(os.path.basename(qry_name[0]))[0]

            # Full paths for query image and label
            query_image_path = os.path.join(image_dir, f'{qry_filename}.jpg')
            query_label_path = os.path.join(label_dir, f'{qry_filename}.png')
            
            # Load original query image for visualization
            query_img_np = cv2.imread(query_image_path)
            
            # Save original query image and label
            shutil.copy(query_image_path, os.path.join(query_folder, f'{qry_filename}_query.jpg'))
            shutil.copy(query_label_path, os.path.join(query_folder, f'{qry_filename}_gt.png'))

            # Extract prediction mask - Ensure it's converted to a 2D array
            if pred_img.dim() == 4:  # This means it's shaped like (N, C, H, W)
                pred_mask = pred_img[0].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
            elif pred_img.dim() == 3:  # This means it's shaped like (C, H, W)
                pred_mask = pred_img.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
            elif pred_img.dim() == 2:  # This is already a 2D array
                pred_mask = pred_img.detach().cpu().numpy().astype(np.uint8)
            else:
                raise ValueError(f"Unexpected prediction image shape: {pred_img.shape}")
            
            # Generate Black Image Overlay
            black_img = np.zeros_like(query_img_np)
            overlayed_black_image = visualize_prediction(black_img, pred_mask)
            cv2.imwrite(os.path.join(query_folder, f'{qry_filename}_pred_black_overlay.png'), overlayed_black_image)
            
            # Generate Original Image Overlay
            overlayed_query_image = visualize_prediction(query_img_np, pred_mask)
            cv2.imwrite(os.path.join(query_folder, f'{qry_filename}_pred_original_overlay.png'), overlayed_query_image)

            # Save the predicted mask
            # cv2.imwrite(os.path.join(query_folder, f'{qry_filename}_pred.png'), pred_mask)

        # Save Support Images
        for way_idx, way in enumerate(supp_imgs):
            for shot_idx, (supp_img, supp_name) in enumerate(zip(way, support_image_names[way_idx])):
                
                # Extract the actual filename (removes path and extension)
                supp_filename = os.path.splitext(os.path.basename(supp_name[0]))[0]

                # Full path for support image
                support_image_path = os.path.join(image_dir, f'{supp_filename}.jpg')
                
                # Copy the original support image to the support folder
                shutil.copy(support_image_path, os.path.join(support_folder, f'{supp_filename}_support.jpg'))


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, gt_mask, episode_num=None, query_image_names=None, support_image_names=None):
        """
        Args:
            supp_imgs: support images`
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config.get('model', {}).get('align', False) and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        ###### New Code Block for Identifying Hard Samples ######
        if self.training and gt_mask is not None:
            # Get config settings
            config = self.config.get('hard_sample_cfg', {})
            conf_threshold = config.get('confidence_threshold', 0.6)
            alpha = config.get('alpha', 1.0)
            beta = config.get('beta', 0.5)
            hardness_threshold = config.get('hardness_threshold', 0.7)
            save_path = config.get('save_path', './hard_samples')
            
            # Calculate the predicted mask and confidence
            pred_mask = torch.argmax(output, dim=1)  # N x H x W
            softmax_output = torch.softmax(output, dim=1)  # N x C x H x W
            confidence_map, _ = torch.max(softmax_output, dim=1)  # N x H x W

            # Calculate misclassification map
            misclassification_map = (pred_mask != gt_mask).float()  # N x H x W

            # Calculate Hardness Score
            hardness_score = (alpha * misclassification_map) + (beta * (1 - confidence_map))
            average_hardness_score = torch.mean(hardness_score)

            # Save hard samples if they meet the threshold
            if average_hardness_score > hardness_threshold:
                self.save_hard_samples(
                    supp_imgs, qry_imgs, output, gt_mask, save_path,
                    query_image_names, support_image_names, img_size
                )
        ###### End of New Code Block ######

        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
