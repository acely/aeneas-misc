#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module implements Voice Activity Detection (VAD) using the Silero VAD library.

The primary class, :class:`~aeneas.vad.VAD`, processes raw audio waveforms to
identify speech and non-speech segments. It returns a boolean mask where ``True``
indicates a speech frame and ``False`` indicates a non-speech frame, based on
the MFCC windowing defined in the RuntimeConfiguration.

Configuration parameters for the Silero VAD model, such as detection threshold,
minimum speech duration, minimum silence duration, window size, and speech padding,
are sourced from the :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
object (rconf) passed to the VAD instance.

This module requires `torch`, `torchaudio`, and `silero-vad` to be installed.
If Silero VAD fails to load, the VAD functionality will be impaired or disabled.

.. versionchanged:: 1.8.0
   Replaced MFCC energy-based VAD with Silero VAD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import torch

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration

# Try to import Silero VAD and its utilities
try:
    torch.hub.set_dir(RuntimeConfiguration.get_default_cache_dir()) # Set cache dir for torch.hub
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False, # Set to True if you always want to download, False for cached
        onnx=False # Set to True if ONNX is preferred
    )
    _get_speech_timestamps = _silero_utils[0]
    _SILERO_VAD_AVAILABLE = True
except Exception as e:
    _SILERO_VAD_AVAILABLE = False
    _silero_model = None
    _get_speech_timestamps = None
    # Log this failure, ideally through the logger, but print for now if logger not avail at module load
    print(f"Failed to load Silero VAD model: {e}")


class VAD(Loggable):
    """
    The voice activity detector (VAD).
    Utilizes the Silero VAD library to perform voice activity detection on raw audio data.

    If the Silero VAD model (`snakers4/silero-vad`) cannot be loaded (e.g., due to missing
    dependencies or network issues during the first download attempt by `torch.hub`),
    VAD functionality will be unavailable, and errors will be logged.

    The VAD process generates a boolean mask corresponding to audio frames (defined by
    the MFCC window shift in `rconf`). This mask indicates speech (`True`) or
    non-speech (`False`) regions.

    Specific Silero VAD parameters (threshold, min_speech_duration_ms, etc.) are
    configured through the `RuntimeConfiguration` object.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"VAD"
    SILERO_MODEL = _silero_model
    GET_SPEECH_TIMESTAMPS = _get_speech_timestamps
    SILERO_VAD_AVAILABLE = _SILERO_VAD_AVAILABLE

    def __init__(self, rconf=None, logger=None):
        super(VAD, self).__init__(rconf=rconf, logger=logger)
        if not self.SILERO_VAD_AVAILABLE:
            self.log(u"Silero VAD model is not available. VAD functionality will be disabled.", error=True)
            # Or raise an error immediately:
            # raise RuntimeError("Silero VAD model failed to load. Please check installation and dependencies.")

    def run_vad(
        self,
        audio_waveform,
        sample_rate,
        # Parameters for Silero VAD, to be fetched from rconf
        # Placeholder names for rconf keys are used below
        vad_threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
        window_size_samples=None,
        speech_pad_ms=None
    ):
        """
        Compute the time intervals containing speech and nonspeech using Silero VAD,
        and return a boolean mask with speech frames set to ``True``,
        and nonspeech frames set to ``False``.

        The Silero VAD parameters (`vad_threshold`, `min_speech_duration_ms`,
        `min_silence_duration_ms`, `window_size_samples`, `speech_pad_ms`)
        are primarily controlled via `RuntimeConfiguration` (rconf). If these
        parameters are passed as ``None`` (the default), their values will be
        fetched from `rconf`. If explicitly provided to this method, those
        values will be used instead.

        The method converts the speech timestamps from Silero VAD (which are in seconds)
        into a frame-based boolean mask. The frame definition (duration) is determined
        by `self.rconf.mws` (MFCC window shift in seconds).

        :param audio_waveform: The raw audio data as a 1D NumPy array or PyTorch Tensor.
                               Multi-channel audio will be converted to mono by using the first channel.
        :type  audio_waveform: :class:`numpy.ndarray` or :class:`torch.Tensor`
        :param int sample_rate: The sample rate of the `audio_waveform` (e.g., 8000, 16000 Hz).
                                Silero VAD supports specific sample rates (e.g., 8kHz, 16kHz).
        :param float vad_threshold: Silero VAD speech probability threshold (e.g., 0.5).
                                    If ``None``, value from `rconf` is used.
        :param int min_speech_duration_ms: Minimum duration for a speech segment in milliseconds.
                                           If ``None``, value from `rconf` is used.
        :param int min_silence_duration_ms: Minimum duration for a silence segment (gap between speech) in milliseconds.
                                            If ``None``, value from `rconf` is used.
        :param int window_size_samples: Window size in samples for Silero VAD analysis (e.g., 512, 1024).
                                        If ``None``, value from `rconf` is used (may depend on sample_rate).
        :param int speech_pad_ms: Padding added to the start and end of detected speech segments in milliseconds.
                                  If ``None``, value from `rconf` is used.
        :rtype: :class:`numpy.ndarray` (1D, boolean)
        :return: A boolean mask where `True` indicates speech and `False` indicates non-speech for each frame.
                 Returns an all-False mask or an empty mask if VAD cannot be performed (e.g., Silero unavailable,
                 invalid parameters).
        """
        if not self.SILERO_VAD_AVAILABLE or self.SILERO_MODEL is None or self.GET_SPEECH_TIMESTAMPS is None:
            self.log(u"Silero VAD is not available or failed to load. Cannot perform VAD.", error=True)
            # Depending on desired behavior, either return an all-False mask or raise error
            # For now, let's assume we need to calculate total_frames for an all-False mask
            # Ensure self.rconf.mws is in seconds. The original code implies it is.
            mfcc_window_shift = self.rconf.mws
            if mfcc_window_shift <= 0: # Guard against division by zero or invalid value
                self.log(u"MFCC window shift (mws) is not positive. Cannot determine mask size for fallback.", error=True)
                return numpy.array([], dtype="bool") # Or a more appropriate fallback
            audio_duration_seconds = len(audio_waveform) / float(sample_rate)
            total_frames = int(round(audio_duration_seconds / mfcc_window_shift))
            return numpy.zeros(total_frames, dtype="bool")

        self.log(u"Computing VAD for wave using Silero VAD")
        # Assuming self.rconf.mws is the MFCC window shift in seconds, as per original VAD code context.
        mfcc_window_shift = self.rconf.mws
        self.log([u"MFCC window shift (s):         %.3f", mfcc_window_shift])

        # Fetch Silero VAD parameters from rconf or use defaults/passed values
        if vad_threshold is None:
            vad_threshold = self.rconf.get(RuntimeConfiguration.VAD_SILERO_THRESHOLD, 0.5) # Example default
            self.log([u"Silero VAD threshold:          %.2f", vad_threshold])
        if min_speech_duration_ms is None:
            min_speech_duration_ms = self.rconf.get(RuntimeConfiguration.VAD_SILERO_MIN_SPEECH_DURATION_MS, 250) # Example default
            self.log([u"Silero VAD min speech (ms):    %d", min_speech_duration_ms])
        if min_silence_duration_ms is None:
            min_silence_duration_ms = self.rconf.get(RuntimeConfiguration.VAD_SILERO_MIN_SILENCE_DURATION_MS, 100) # Example default
            self.log([u"Silero VAD min silence (ms):   %d", min_silence_duration_ms])
        if window_size_samples is None:
            # Silero VAD default values: 512, 1024, 1536 for 16kHz; 256, 512, 768 for 8kHz.
            # This should ideally be chosen based on sample_rate.
            default_window_size = 512 if sample_rate == 8000 else 1024 # Simplified default
            window_size_samples = self.rconf.get(RuntimeConfiguration.VAD_SILERO_WINDOW_SIZE_SAMPLES, default_window_size)
            self.log([u"Silero VAD window size (samp): %d", window_size_samples])
        if speech_pad_ms is None:
            speech_pad_ms = self.rconf.get(RuntimeConfiguration.VAD_SILERO_SPEECH_PAD_MS, 30) # Example default
            self.log([u"Silero VAD speech pad (ms):    %d", speech_pad_ms])

        # Ensure audio_waveform is a PyTorch tensor
        if not isinstance(audio_waveform, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_waveform)
        else:
            audio_tensor = audio_waveform
        
        if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1: # Mono channel but in shape (1, N)
            audio_tensor = audio_tensor.squeeze(0)
        elif audio_tensor.ndim > 1:
            self.log(u"Audio waveform has multiple channels. Using the first channel for VAD.", warning=True)
            audio_tensor = audio_tensor[0] # Assuming first channel

        # Call Silero VAD
        self.log(u"Running Silero VAD model...")
        speech_timestamps = self.GET_SPEECH_TIMESTAMPS(
            audio_tensor,
            self.SILERO_MODEL,
            sample_rate=sample_rate,
            return_seconds=True,  # Important for mask calculation
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms
        )
        self.log(u"Silero VAD model processing complete.")
        self.log([u"Detected %d speech segments", len(speech_timestamps)])

        # Convert timestamps to boolean mask
        audio_duration_seconds = len(audio_waveform) / float(sample_rate)
        if mfcc_window_shift <= 0:
            self.log(u"MFCC window shift is not positive. Cannot create frame mask.", error=True)
            # Fallback: or calculate total_frames based on a default if possible, or raise error
            # For now, return an all-false mask of arbitrary or zero length if this occurs.
            # This indicates a configuration problem.
            return numpy.array([], dtype="bool")

        total_frames = int(round(audio_duration_seconds / mfcc_window_shift))
        self.log([u"Audio duration (s):            %.3f", audio_duration_seconds])
        self.log([u"Total frames for mask:         %d", total_frames])
        
        mask = numpy.zeros(total_frames, dtype="bool")

        if not speech_timestamps:
            self.log(u"No speech segments detected by Silero VAD.")
        else:
            for segment in speech_timestamps:
                start_time_s = segment['start']
                end_time_s = segment['end']
                
                start_frame = int(start_time_s / mfcc_window_shift)
                end_frame = int(end_time_s / mfcc_window_shift)

                # Ensure indices are within bounds
                start_frame = max(0, start_frame)
                end_frame = min(total_frames, end_frame)
                
                if start_frame < end_frame: # Ensure start_frame is strictly less than end_frame
                    mask[start_frame:end_frame] = True
                elif start_frame == end_frame and start_frame < total_frames: # Handle very short segments that fall into one frame
                     mask[start_frame] = True


            # Log a summary of speech coverage
            speech_frames = numpy.sum(mask)
            self.log([u"Number of speech frames:       %d (%.2f%%)", speech_frames, (speech_frames / float(total_frames) * 100) if total_frames > 0 else 0.0])

        self.log(u"VAD mask computation complete.")
        return mask
