from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Grid_prob(Hook):

    def __init__(self,
                 num_last_epochs=15,
                 skip_type_keys=('Mosaic', 'RandomAffine', 'MixUp', 'RMosaic', 'RMixUp')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            # runner.logger.info('Add additional L1 loss now!')
            # model.bbox_head.use_l1 = True