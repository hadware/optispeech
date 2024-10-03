import torch
from omegaconf import DictConfig
from torch import nn
from torch.nested import nested_tensor

from optispeech.utils import sequence_mask
from optispeech.values import InferenceInputs, InferenceOutputs
from .base_lightning_module import BaseLightningModule
from .generator.generator import InferenceOutput
from .generator.wavenext import WaveNeXt


class OptiSpeech(BaseLightningModule):
    def __init__(
            self,
            dim: int,
            generator: nn.Module,
            vocoder: nn.Module,
            train_args: DictConfig,
            data_args: DictConfig,
            inference_args: DictConfig,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Sanity checks
        if (train_args.gradient_accumulate_batches is not None) and (train_args.gradient_accumulate_batches <= 0):
            raise ValueError("gradient_accumulate_batches should be a positive number")

        if data_args.num_speakers < 1:
            raise ValueError("num_speakers should be a positive integer >= 1")

        self.train_args = train_args
        self.data_args = data_args
        self.inference_args = inference_args

        self.text_processor = self.data_args.text_processor

        self.num_speakers = data_args.num_speakers
        self.sample_rate = data_args.feature_extractor.sample_rate
        self.hop_length = data_args.feature_extractor.hop_length

        # GAN training requires this
        self.automatic_optimization = False

        self.generator = generator(
            dim=dim,
            feature_extractor=data_args.feature_extractor,
            data_statistics=data_args.data_statistics,
            num_speakers=self.data_args.num_speakers,
            num_languages=self.text_processor.num_languages,
        )

        self.vocoder = vocoder

    @torch.inference_mode()
    def synthesise(self, inputs: InferenceInputs) -> InferenceOutputs:
        inputs = inputs.as_torch()
        inputs = inputs.to(self.device)
        synth_outputs: InferenceOutput = self.generator.synthesise(
            x=inputs.x,
            x_lengths=inputs.x_lengths.to("cpu"),
            sids=inputs.sids,
            d_factor=inputs.d_factor,
            p_factor=inputs.p_factor,
            e_factor=inputs.e_factor
        )

        y_lengths = synth_outputs["durations"].sum(dim=1)
        wav_lengths = y_lengths * self.hop_length

        if isinstance(self.vocoder, WaveNeXt):
            y_max_length = y_lengths.max()
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(x)
            target_padding_mask = ~y_mask.squeeze(1).bool()

            # Generate wav
            batched_wav = self.vocoder(synth_outputs["mel_hat"].transpose(1, 2), target_padding_mask)

        else: # it's a melgan/hifigan
            wavs = []
            # TODO: check
            for i, y_length in enumerate(y_lengths):
                mel = synth_outputs["mel_hat"][i, :y_length, :]
                wav = self.vocoder.infer(mel)
                wavs.append(wav)
            batched_wav = nested_tensor(wavs, device=self.device).to_padded_tensor(0)

        return InferenceOutputs(
            wav=batched_wav,
            wav_lengths=wav_lengths,
            durations=synth_outputs["durations"],
            pitch=synth_outputs["pitch"],
            energy=synth_outputs["energy"],
            am_rtf=synth_outputs["am_rtf"],
        )

    def prepare_input(
            self,
            text: str,
            *,
            language: str | None = None,
            speaker: str | int | None = None,
            d_factor: float = None,
            p_factor: float = None,
            e_factor: float = None,
            split_sentences: bool = True,
    ) -> InferenceInputs:
        """
        Convenient helper.

        Args:
            text (str): input text
            language (str|None): language of input text
            speaker (int|str|None): speaker name
            d_factor (float|None): scaling value for duration
            p_factor (float|None): scaling value for pitch
            e_factor (float|None): scaling value for energy
            split_sentences (bool): split text into sentences (each sentence is an element in the batch)

        Returns:
            InferenceInputs
        """
        sid = None
        if self.num_speakers > 1:
            if speaker is None:
                sid = 0
            elif type(speaker) is str:
                try:
                    sid = self.speakers.index(speaker)
                except IndexError:
                    raise ValueError(f"A speaker with the given name `{speaker}` was not found in speaker list")
            elif type(speaker) is int:
                sid = speaker

        input_ids, clean_text = self.text_processor(text, lang=language, split_sentences=split_sentences)
        if split_sentences:
            lengths = [len(phids) for phids in input_ids]
        else:
            lengths = [len(input_ids)]
            input_ids = [input_ids]

        sids = [sid] * len(input_ids) if sid is not None else None

        inputs = InferenceInputs.from_ids_and_lengths(
            ids=input_ids,
            lengths=lengths,
            clean_text=clean_text,
            sids=sids,
            d_factor=d_factor or self.inference_args.d_factor,
            p_factor=p_factor or self.inference_args.p_factor,
            e_factor=e_factor or self.inference_args.e_factor
        )
        inputs = inputs.as_torch()
        inputs = inputs.to(self.device)
        return inputs
