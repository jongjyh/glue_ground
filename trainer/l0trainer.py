from transformers.trainer import *


class l0trainer(Trainer):
    pass
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            arch_parameters = [name for name,_ in self.model.named_parameters() if 'qz_loga' in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n not in arch_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in arch_parameters],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in arch_parameters],
                    "weight_decay":0.1,
                    "lr":self.model.config.a_lr,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    def evaluation_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> EvalLoopOutput:
        eval_outputs = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        p = torch.asarray([k[2] for k in self.model.get_flops()]).mean()
        eval_outputs.metrics['eval_p']=p.detach().item()
        return eval_outputs 


    # def evaluate(
    #     self,
    #     eval_dataset: Optional[Dataset] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> Dict[str, float]:
    #     """
    #     Run evaluation and returns metrics.

    #     The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
    #     (pass it to the init `compute_metrics` argument).

    #     You can also subclass and override this method to inject custom behavior.

    #     Args:
    #         eval_dataset (`Dataset`, *optional*):
    #             Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
    #             accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
    #             method.
    #         ignore_keys (`Lst[str]`, *optional*):
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions.
    #         metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
    #             An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
    #             "eval_bleu" if the prefix is "eval" (default)

    #     Returns:
    #         A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
    #         dictionary also contains the epoch number which comes from the training state.
    #     """
    #     # memory metrics - must set up as early as possible
    #     self._memory_tracker.start()

    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     start_time = time.time()

    #     eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    #     output = eval_loop(
    #         eval_dataloader,
    #         description="Evaluation",
    #         # No point gathering the predictions if there are no metrics, otherwise we defer to
    #         # self.args.prediction_loss_only
    #         prediction_loss_only=True if self.compute_metrics is None else None,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix,
    #     )

    #     total_batch_size = self.args.eval_batch_size * self.args.world_size
    #     output.metrics.update(
    #         speed_metrics(
    #             metric_key_prefix,
    #             start_time,
    #             num_samples=output.num_samples,
    #             num_steps=math.ceil(output.num_samples / total_batch_size),
    #         )
    #     )
    #     # output.metrics['FLOPs']=self.model.get_flops()
    #     self.log(output.metrics)

    #     if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #         # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #         xm.master_print(met.metrics_report())

    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

    #     self._memory_tracker.stop_and_update_metrics(output.metrics)

    #     return output.metrics
