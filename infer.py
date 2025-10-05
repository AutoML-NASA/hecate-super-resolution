import logging
import torch
from os import path as osp
import time

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        num_imgs = len(test_loader.dataset)
        logger.info(f'Testing {test_set_name}...')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img']
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0  # seconds

        ms_per_img = (elapsed / max(1, num_imgs)) * 1000.0
        fps = num_imgs / elapsed if elapsed > 0 else float('inf')
        logger.info(
            f"[{test_set_name}] End-to-end time: {elapsed:.3f}s, "
            f"{ms_per_img:.2f} ms/img, {fps:.2f} img/s"
        )

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)