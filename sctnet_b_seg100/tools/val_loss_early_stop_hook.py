from contextlib import nullcontext

from mmcv.parallel import scatter
from mmcv.runner import HOOKS, Hook
from mmseg.datasets import build_dataloader, build_dataset


@HOOKS.register_module()
class ValLossEarlyStopHook(Hook):
    def __init__(self, dataset_cfg, interval=1, samples_per_gpu=1, workers_per_gpu=2):
        self.dataset_cfg = dataset_cfg
        self.interval = interval
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.val_dataloader = None

    def before_run(self, runner):
        dataset = build_dataset(self.dataset_cfg)
        self.val_dataloader = build_dataloader(
            dataset,
            samples_per_gpu=self.samples_per_gpu,
            workers_per_gpu=self.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False,
            seed=None,
            drop_last=False,
            persistent_workers=False,
        )
        runner.logger.info('[ValLossEarlyStopHook] Built val-loss dataloader.')

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        model = runner.model
        was_training = model.training
        model.eval()

        device = next(model.parameters()).device
        loss_values = []

        for data in self.val_dataloader:
            if device.type == 'cuda':
                gpu_id = 0 if device.index is None else int(device.index)
                data = scatter(data, [gpu_id])[0]

            sync_ctx = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
            with sync_ctx:
                outputs = model(return_loss=True, **data)

            if isinstance(outputs, dict):
                total_loss = 0.0

                for key, value in outputs.items():
                    if 'loss' not in key.lower():
                        continue

                    if isinstance(value, (list, tuple)):
                        for v in value:
                            if hasattr(v, 'mean'):
                                total_loss = total_loss + v.mean()
                    else:
                        if hasattr(value, 'mean'):
                            total_loss = total_loss + value.mean()

                if hasattr(total_loss, 'item'):
                    loss_values.append(float(total_loss.item()))
                else:
                    loss_values.append(float(total_loss))
            else:
                loss_values.append(float(outputs.item() if hasattr(outputs, 'item') else outputs))

        mean_loss = sum(loss_values) / max(len(loss_values), 1)

        runner.log_buffer.output['val_loss'] = float(mean_loss)
        runner.log_buffer.ready = True

        runner.logger.info(
            f'[ValLossEarlyStopHook] epoch={runner.epoch + 1} val_loss(mean)={mean_loss:.6f}'
        )

        if was_training:
            model.train()
