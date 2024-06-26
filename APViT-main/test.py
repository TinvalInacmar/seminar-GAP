import argparse
import os
import warnings
 
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
 
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.core import wrap_fp16_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--subset', default='test', help='subset to test')
    parser.add_argument('--store', default=True, type=bool, help='store test results and misaligned results in csv files')
    parser.add_argument(
        '--metric', type=str, default='accuracy', help='evaluation metric')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
 
 
def main():
    args = parse_args()
 
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
 
    # cfg.model.extractor.pretrained = None
    # cfg.model.vit.pretrained = None
 
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
 
    # build the dataloader
    dataset = build_dataset(getattr(cfg.data, args.subset))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)
 
    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    if args.checkpoint:    
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
 
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
 
    rank, _ = get_dist_info()
    if rank == 0:
        if args.metric == 'save_result':
            scores = np.vstack(outputs)
            pred_label = np.argmax(scores, axis=1)
            gt_label = dataset.get_gt_labels()
            assert len(pred_label) == len(gt_label)
            result = []
            for i, info in enumerate(dataset.data_infos):
                file_name = info['img_info']['filename']
                result.append(f'{file_name}, {gt_label[i]==pred_label[i]}')
            with open('output/save_result.csv', 'w') as f:
                f.write('\n'.join(result))
            exit()
 
        if args.metric != '':
            results = dataset.evaluate(outputs, args.metric, store_results=args.store)
            for topk, acc in results.items():
                print(f'\n{topk} accuracy: {acc:.2f}')
        else:
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            # if 'CLASSES' in checkpoint['meta']:
            #     CLASSES = checkpoint['meta']['CLASSES']
            from mmcls.datasets import ImageNet
            from mmcls.datasets import affectnet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                            'meta data, use imagenet by default.')
            CLASSES = affectnet.DATASET_CLASSES
            pred_class = [CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not args.out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)
 
 
if __name__ == '__main__':
    main()