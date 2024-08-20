from typing import List, Optional

import torch
from torch import nn

from optispeech.utils import pad_list
from optispeech.values import InferenceInputs, InferenceOutputs

from .base_lightning_module import BaseLightningModule


class OptiSpeech(BaseLightningModule):
    def __init__(
        self,
        dim,
        generator,
        discriminator,
        train_args,
        data_args,
        inference_args,
        optimizer=None,
        scheduler=None,
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
        self.discriminator = discriminator(feature_extractor=data_args.feature_extractor)

    @torch.inference_mode()
    def synthesise(self, inputs: InferenceInputs) -> InferenceOutputs:
        inputs = inputs.as_torch()
        inputs = inputs.to(self.device)

        synth_outputs = self.generator.synthesise(
            x=inputs.x,
            x_lengths=inputs.x_lengths.to("cpu"),
            sids=inputs.sids,
            lids=inputs.lids,
            d_factor=inputs.d_factor,
            p_factor=inputs.p_factor,
            e_factor=inputs.e_factor
        )
        return InferenceOutputs(
            wav=synth_outputs["wav"],
            wav_lengths=synth_outputs["wav_lengths"],
            durations=synth_outputs["durations"],
            pitch=synth_outputs["pitch"],
            energy=synth_outputs["energy"],
            latency=synth_outputs["latency"],
            rtf=synth_outputs["rtf"],
            am_rtf=synth_outputs["am_rtf"],
            v_rtf=synth_outputs["v_rtf"],
        )

    def prepare_input(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker: str | int | None = None,
        d_factor: float=None, 
        p_factor: float=None,
        e_factor: float=None,
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
        languages = self.text_processor.languages
        if language is None:
            language = languages[0]
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
        else:
            sid = None
        if self.text_processor.is_multi_language:
            try:
                lid = languages.index(language)
            except IndexError:
                raise ValueError(f"A language with the given name `{language}` was not found in language list")
        else:
            lid = None

        phoneme_ids, clean_text = self.text_processor(text, lang=language, split_sentences=split_sentences)
        if split_sentences:
            x_lengths = torch.LongTensor([len(phids) for phids in phoneme_ids])
            x = pad_list([torch.LongTensor(phids) for phids in phoneme_ids], pad_value=0)
        else:
            x_lengths = torch.LongTensor([1])
            x = torch.LongTensor(phoneme_ids).unsqueeze(0)

        sids = [sid] * x.shape[0] if sid is not None else None
        lids = [lid] * x.shape[0] if lid is not None else None

        return InferenceInputs(
            clean_text=clean_text,
            x=x.long(),
            x_lengths=x_lengths.long(),
            sids=sids,
            lids=lids,
            d_factor=d_factor or self.inference_args.d_factor,
            p_factor=p_factor or self.inference_args.p_factor,
            e_factor=e_factor or self.inference_args.e_factor
        )
