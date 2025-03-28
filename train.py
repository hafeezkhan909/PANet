"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
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
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    hard_samples_found = 0  # Track number of hard samples encountered
    reused_loss = {'loss': 0, 'align_loss': 0}  # Track loss for reused hard samples
    reused_sample_count = 0
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Extract image names from the batch
        query_image_names = sample_batched['query_filenames']  # This is already returned by the DataLoader
        support_image_names = sample_batched['support_filenames']

        # Forward and Backward
        optimizer.zero_grad()
        # query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
        #                                query_images)
        query_pred, align_loss, hard_sample_detected = model(
            support_images, support_fg_mask, support_bg_mask, 
            query_images, gt_mask=query_labels,
            query_image_names=query_image_names,
            support_image_names=support_image_names
        )


        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        # query_loss = query_loss.detach().data.cpu().numpy()
        # align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        # _run.log_scalar('loss', query_loss)
        # _run.log_scalar('align_loss', align_loss)
        # # log_loss['loss'] += query_loss
        # # log_loss['align_loss'] += align_loss
        # log_loss['loss'] += query_loss.item()
        # log_loss['align_loss'] += align_loss if isinstance(align_loss, float) else align_loss.item()
        # Log loss - STANDARD TRAINING LOOP
        query_loss_value = query_loss.detach().cpu().item()
        align_loss_value = align_loss.detach().cpu().item() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss_value)
        _run.log_scalar('align_loss', align_loss_value)
        log_loss['loss'] += query_loss_value
        log_loss['align_loss'] += align_loss_value
        loss = log_loss['loss'] / (i_iter + 1)
        align_loss = log_loss['align_loss'] / (i_iter + 1)
        print(f'SAMPLE: step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')
        # If a hard sample was identified, reuse it immediately
        if hard_sample_detected:
            # Log loss BEFORE reuse for comparison purposes
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'BEFORE REUSING THE SAMPLE: step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')
            
            # Increment hard sample counter
            hard_samples_found += 1
            reused_sample_count += 1  # Count reused samples separately, NOT by modifying i_iter
            print(reused_sample_count)
            # Reuse the same support-query pairs
            optimizer.zero_grad()
            # Convert tensors to lists of lists before reuse
            query_pred, align_loss, _ = model(
                support_images, support_fg_mask, support_bg_mask, 
                query_images, gt_mask=query_labels,
                query_image_names=query_image_names,
                support_image_names=support_image_names
            )

            # Compute loss for reused sample
            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss * _config['align_loss_scaler']
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log the loss after reuse
            query_loss_value = query_loss.detach().cpu().item()
            align_loss_value = align_loss.detach().cpu().item() if align_loss != 0 else 0
            # Update the log_loss dictionary to reflect the new, reused loss
            log_loss['loss'] = log_loss['loss'] - (log_loss['loss'] / (i_iter + 1)) + query_loss_value
            log_loss['align_loss'] = log_loss['align_loss'] - (log_loss['align_loss'] / (i_iter + 1)) + align_loss_value

            # Logging the reused sample losses separately
            _run.log_scalar('reused_loss', query_loss_value)
            _run.log_scalar('reused_align_loss', align_loss_value)
            reused_loss['loss'] += query_loss_value
            reused_loss['align_loss'] += align_loss_value

            # Report the loss after reuse
            print(f'AFTER REUSING THE SAMPLE: Reused Sample Count: {reused_sample_count}, Reused Loss: {query_loss_value}, Reused Align Loss: {align_loss_value}')


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
    
    _log.info('###### Training Summary ######')
    total_steps = i_iter + 1
    avg_loss = log_loss['loss'] / total_steps
    avg_align_loss = log_loss['align_loss'] / total_steps

    if reused_sample_count > 0:
        avg_reused_loss = reused_loss['loss'] / reused_sample_count
        avg_reused_align_loss = reused_loss['align_loss'] / reused_sample_count
    else:
        avg_reused_loss = avg_reused_align_loss = 0

    print(f"\nTraining Completed! Total Steps: {total_steps}")
    print(f"Average Loss: {avg_loss:.4f}, Average Align Loss: {avg_align_loss:.4f}")
    print(f"Hard Samples Reused: {reused_sample_count}")
    print(f"Average Reused Loss: {avg_reused_loss:.4f}, Average Reused Align Loss: {avg_reused_align_loss:.4f}")

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))