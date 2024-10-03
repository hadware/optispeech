"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
import math
from abc import ABC
from typing import Any, Dict

import numpy as np
import torch
import torchaudio
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from optispeech.model.generator.generator import GeneratorTrainingOutput
from optispeech.utils import get_pylogger, plot_attention, plot_tensor
from optispeech.utils.segments import get_segments, get_segments_numpy, get_random_segments

log = get_pylogger(__name__)


class BaseLightningModule(LightningModule, ABC):
    def _process_batch(self, batch) -> GeneratorTrainingOutput:
        sids = batch["sids"]
        return self.generator(
            x=batch["x"].to(self.device),
            x_lengths=batch["x_lengths"].to(self.device),
            mel=batch["mel"].to(self.device),
            mel_lengths=batch["mel_lengths"].to(self.device),
            pitches=batch["pitches"].to(self.device),
            energies=batch["energies"].to(self.device),
            sids=sids.to(self.device) if sids is not None else None,
        )

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        opt_gen = self.hparams.optimizer(gen_params)

        # Max steps per optimizer
        max_steps = self.trainer.max_steps // 2
        # Adjust by gradient accumulation batches
        if self.train_args.gradient_accumulate_batches is not None:
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else -1
            max_steps = math.ceil(max_steps / self.train_args.gradient_accumulate_batches) * max(max_epochs, 1)

        if "num_training_steps" in self.hparams.scheduler.keywords:
            self.hparams.scheduler.keywords["num_training_steps"] = max_steps
        scheduler_gen = self.hparams.scheduler(opt_gen, last_epoch=getattr("self", "ckpt_loaded_epoch", -1))
        return (
            [opt_gen],
            [{"scheduler": scheduler_gen, "interval": "step"}],
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

    def training_step(self, batch, batch_idx, **kwargs):
        # manual gradient accumulation
        gradient_accumulate_batches = self.train_args.gradient_accumulate_batches
        if gradient_accumulate_batches is not None:
            loss_scaling_factor = float(gradient_accumulate_batches)
            should_apply_gradients = (batch_idx + 1) % gradient_accumulate_batches == 0
        else:
            loss_scaling_factor = 1.0
            should_apply_gradients = True

        # Extract generator/discriminator optimizer/scheduler
        opt_g = self.optimizers()
        sched_g = self.lr_schedulers()
        # train generator
        self.toggle_optimizer(opt_g)
        loss_g = self.training_step_g(batch)
        # Scale (grad accumulate)
        loss_g /= loss_scaling_factor
        self.manual_backward(loss_g)
        if should_apply_gradients:
            self.clip_gradients(
                opt_g, gradient_clip_val=self.train_args.gradient_clip_val, gradient_clip_algorithm="norm"
            )
            opt_g.step()
            sched_g.step()
            opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def training_step_g(self, batch):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_am_loss = gen_outputs["loss"]
        log_outputs.update(
            {
                "total_loss/train_am_loss": gen_am_loss.item(),
                "gen_subloss/train_alighn_loss": gen_outputs["align_loss"].item(),
                "gen_subloss/train_duration_loss": gen_outputs["duration_loss"].item(),
                "gen_subloss/train_pitch_loss": gen_outputs["pitch_loss"].item(),
                "gen_subloss/train_energy_loss": gen_outputs["energy_loss"].item(),
            }
        )

        gen_adv_loss = 0.0

        loss = gen_am_loss + gen_adv_loss
        log_outputs["total_loss/generator"] = loss.item()
        log_dict = {}
        for (name, value) in log_outputs.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.detach().cpu()
            else:
                log_dict[name] = value
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size,
        )
        return loss


    def on_validation_epoch_start(self):
        if self.train_args.evaluate_utmos:
            from optispeech.vendor.metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_am_loss = gen_outputs["loss"]
        log_outputs.update(
            {
                "total_loss/val_am_loss": gen_outputs["loss"].item(),
                "gen_subloss/val_alighn_loss": gen_outputs["align_loss"].item(),
                "gen_subloss/val_duration_loss": gen_outputs["duration_loss"].item(),
                "gen_subloss/val_pitch_loss": gen_outputs["pitch_loss"].item(),
                "gen_subloss/val_energy_loss": gen_outputs["energy_loss"].item()
            }
        )
        total_loss = gen_am_loss
        log_outputs["total_loss/val_total"] = total_loss.item()
        log_dict = {}
        for (name, value) in log_outputs.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.detach().cpu()
            else:
                log_dict[name] = value
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size,
        )

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    gt_wav = one_batch["wav"][i].squeeze()
                    self.logger.experiment.add_audio(
                        f"wav/original_{i}", gt_wav, self.global_step, self.sample_rate
                    )
                    mel = one_batch["mel"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"mel/original_{i}",
                        plot_tensor(mel.detach().squeeze().float().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                synth_out = self.generator.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                wav_hat: torch.Tensor = self.vocoder(synth_out["mel"].transpose(1,2))
                mel_hat = self.data_args.feature_extractor.get_mel(wav_hat.squeeze().cpu().numpy())
                self.logger.experiment.add_image(
                    f"mel/decoded_{i}",
                    plot_tensor(synth_out["mel"].transpose(1,2).squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"mel/generated_{i}",
                    plot_tensor(mel_hat.squeeze()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_audio(f"wav/generated_{i}", wav_hat.squeeze(), self.global_step, self.sample_rate)

    def test_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed with respect to `gradient_accumulate_batches`
        """
        if self.train_args.gradient_accumulate_batches is not None:
            global_step = self.trainer.fit_loop.total_batch_idx // self.train_args.gradient_accumulate_batches
        else:
            global_step = self.trainer.fit_loop.total_batch_idx
        return int(global_step)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init
