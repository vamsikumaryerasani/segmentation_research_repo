from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """
    Early stopping based on a scalar logged in runner.log_buffer.output, e.g. 'val.loss'.

    Stops training by setting runner._max_epochs to current_epoch+1 (MMCV EpochBasedRunner stops cleanly).
    """

    def __init__(self,
                 monitor='val.loss',
                 rule='less',           # 'less' for loss, 'greater' for metrics
                 patience=5,
                 min_delta=0.0,
                 start_epoch=1,
                 interval=1):
        self.monitor = monitor
        self.rule = rule
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.start_epoch = int(start_epoch)
        self.interval = int(interval)

        self.best = None
        self.num_bad = 0

    def _is_improved(self, current):
        if self.best is None:
            return True

        if self.rule == 'less':
            return current < (self.best - self.min_delta)
        elif self.rule == 'greater':
            return current > (self.best + self.min_delta)
        else:
            raise ValueError("rule must be 'less' or 'greater'")

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1  # 1-based for humans

        if epoch < self.start_epoch:
            return
        if (epoch % self.interval) != 0:
            return

        current = runner.log_buffer.output.get(self.monitor, None)
        if current is None:
            runner.logger.warning(
                f'[EarlyStoppingHook] "{self.monitor}" not found in log_buffer.output. '
                f'Make sure your val-loss hook writes runner.log_buffer.output["{self.monitor}"].'
            )
            return

        current = float(current)

        if self._is_improved(current):
            self.best = current
            self.num_bad = 0
            runner.logger.info(f'[EarlyStoppingHook] improved {self.monitor}={current:.6f}')
        else:
            self.num_bad += 1
            runner.logger.info(f'[EarlyStoppingHook] no improvement: {self.num_bad}/{self.patience}')

            if self.num_bad >= self.patience:
                runner.logger.info('[EarlyStoppingHook] Early stopping triggered.')
                # stop cleanly after this epoch
                runner._max_epochs = runner.epoch + 1
