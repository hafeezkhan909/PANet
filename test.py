"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize
# from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex
import cv2

def visualize_prediction(image, prediction):
    """
    Overlay the segmentation prediction on the image.

    Args:
        image (np.ndarray): Original query image (H x W x 3).
        prediction (np.ndarray): Predicted segmentation map (H x W).

    Returns:
        np.ndarray: Image with prediction overlay.
    """
   
    color_map = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]])  # Example color map
    overlay = color_map[prediction]  # Map prediction to colors
    # print(f"Image shape: {image.shape}, Overlay shape: {overlay.shape}")
    overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))  # Resize to (800, 320)
    blended = cv2.addWeighted(image, 0.7, overlay.astype(np.uint8), 0.3, 0)  # Blend
    return blended

def load_and_resize_image(image_path, size=(320, 800)):
    """
    Load an image from file and resize it to the given size.
    
    Args:
        image_path (str): Path to the image file.
        size (int): Desired size for both width and height.

    Returns:
        np.ndarray: Resized image in (H, W, C) format.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Error: Could not load image {image_path}")

    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)
    # print(img.shape)
    return img


@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    output_dir = os.path.join(_run.observers[0].dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)  # Create the main predictions directory.

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]

    # if _config['scribble_dilation'] > 0:
    #     transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            # Create a directory for the current run
            run_dir = os.path.join(output_dir, f'run_{run + 1}')
            os.makedirs(run_dir, exist_ok=True)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['n_steps'] * _config['batch_size'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries']
            )
            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")


            for batch_idx, sample_batched in enumerate(tqdm.tqdm(testloader)):
                batch_dir = os.path.join(run_dir, f'batch_{batch_idx + 1}')
                os.makedirs(batch_dir, exist_ok=True)  # Create directory for each batch.

                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])

                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                support_images_filenames = sample_batched['support_filenames']
                

                suffix = 'scribble' if _config['scribble'] else 'mask'

                if _config['bbox']:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                
                query_images_filenames = sample_batched['query_filenames']

                query_labels = torch.cat(
                    [query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)
                
                # Save query images using actual file paths
                for i, query_filename in enumerate(query_images_filenames):
                    query_filename = query_images_filenames[i][0]  # Extract string from list
                    # print(f"Loading image: {query_filename}")  # Debugging
                    query_img_np = load_and_resize_image(query_images_filenames[i][0])

                    if query_img_np is None:
                        print(f"Error: Could not read image at {query_filename}")
                    query_pred_np = query_pred[i].argmax(dim=0).cpu().numpy()  # H x W
                    overlay = visualize_prediction(query_img_np, query_pred_np)

                    query_dir = os.path.join(batch_dir, f'query_{i + 1}')
                    os.makedirs(query_dir, exist_ok=True)

                    cv2.imwrite(os.path.join(query_dir, 'query_image.png'), query_img_np)
                    cv2.imwrite(os.path.join(query_dir, 'query_prediction.png'), overlay)

                    # Save support images for this query
                    support_dir = os.path.join(query_dir, 'support_images')
                    os.makedirs(support_dir, exist_ok=True)

                    for way_idx, way_filenames in enumerate(support_images_filenames):
                        way_dir = os.path.join(support_dir, f'way_{way_idx + 1}')
                        os.makedirs(way_dir, exist_ok=True)

                        for shot_idx, support_filename in enumerate(way_filenames):
                            support_img_np = load_and_resize_image(support_images_filenames[way_idx][shot_idx][0])
                            cv2.imwrite(os.path.join(way_dir, f'shot_{shot_idx + 1}.png'), support_img_np)

                # Metric recording (unchanged)
                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')