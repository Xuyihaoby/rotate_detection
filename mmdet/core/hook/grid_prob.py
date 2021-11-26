from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Grid_prob(Hook):

    def __init__(self):
        pass

    def before_train_epoch(self, runner):
        pipeline = runner.data_loader.dataset.pipeline.transforms
        assert hasattr(pipeline[2],'response')
        grid = pipeline[2]
        grid.set_prob(runner.epoch, runner.max_epochs)
        runner.logger.info('change prob')
