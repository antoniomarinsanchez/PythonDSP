"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: MIT License (https://opensource.org/licenses/MIT)
"""
import math

import numpy as np
import matplotlib.pyplot as plt

# Constants
PI2 = math.pi * 2


class Wave:
    """Represents a discrete_time waveform."""

    def __init__(self, ys, ts=None, framerate=None):
        """Initialize the wave.

        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        # posibly asarray()
        self.ys = np.asanyarray(ys)
        self.framerate = framerate if framerate is not None else 11025

        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            # posibly asarray()
            self.ts = np.asanyarray(ts)

    def __len__(self):
        return len(self.ys)

    @property
    def start(self):
        return self.ts[0]

    @property
    def end(self):
        return self.ts[-1]

    def find_index(self, t):
        """Find the index corresponding to a given time."""
        n = len(self)
        start = self.start
        end = self.end
        i = round((n - 1) * (t - start) / (end - start))
        return int(i)

    def slice(self, i, j):
        """Makes a slice for a Wave

        i: first slice index
        j: second slice index
        """

        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Wave(ys, ts, self.framerate)

    def segment(self, start=None, duration=None):
        """Extracts a segment.

        start: float start time in seconds
        duration: float duration in seconds

        returns: Wave
        """
        if start is None:
            start = self.ts[0]
            i = 0
        else:
            i = self.find_index(start)

        j = None if duration is None else self.find_index(start + duration)
        return self.slice(i, j)

    def get_xfactor(self, options):
        """Extracts xfactor for plotting purposes"""
        try:
            xfactor = options["xfactor"]
            options.pop("xfactor")
        except KeyError:
            xfactor = 1
        return xfactor

    def plot(self, **options):
        """Plots the wave.

        It the ys are complex, plots the real part.

        """
        xfactor = self.get_xfactor(options)
        plt.plot(self.ts * xfactor, np.real(self.ys), **options)


class Signal:
    """Represent a time-varying signal"""

    def __add__(self, other):
        """Adds two signals.

        other: Signal

        return: Signal
        """
        if other == 0:
            return self
        return SumSignal(self, other)

    # commutative property
    __radd__ = __add__

    @property
    def period(self):
        """Period of the signal in seconds (property).

        Since this is used primarily for purposes of plotting,
        the default behavior is to return a value, 0.1 seconds,
        that is reasonable for many signals.

        returns: float seconds
        """
        return 0.1

    def plot(self, framerate=11025):
        """Plots the signal.

        The default behavior is to plot three periods.

        framerate: sample per seconds
        """
        duration = round(self.period * 3)
        wave = self.make_wave(duration, start=0, framerate=framerate)
        wave.plot()

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        n = round(duration * framerate)
        ts = start + np.arange(n) / framerate
        ys = self.evaluate(ts)
        return Wave(ys, ts, framerate=framerate)


class SumSignal(Signal):
    """Represents the sum of signals."""

    def __init__(self, *args):
        """Initializes the sum.

        args: tuple of signals
        """
        self.signals = args

    @property
    def period(self):
        """Period of the signal in seconds.

        Note: this is not correct; it's mostly a placekeeper.

        But it is correct for a harmonic sequence where all
        component frequencies are multiples of the fundamental.

        returns: float seconds
        """
        return max(sig.period for sig in self.signals)

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        return sum(sig.evaluate(ts) for sig in self.signals)


class Sinusoid(Signal):
    """Represents a sinusoidal signal."""

    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        """Initializes a sinusoidal signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 in nominal max
        offset: float phase offset in radians
        func: function that maps phase to amplitude
        """
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    @property
    def period(self):
        """Period of the signal in seconds (property).

        returns: float seconds
        """
        return 1.0 / self.freq

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float seconds
        """
        ts = np.asarray(ts)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys


def cos_signal(freq=440, amp=10, offset=0):
    """Makes a cosine sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 in nominal max
    offset: float phase offset in radians

    returns: Sinusoid object"""
    return Sinusoid(freq, amp, offset, func=np.cos)


def sin_signal(freq=440, amp=10, offset=0):
    """Makes a cosine sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 in nominal max
    offset: float phase offset in radians

    returns: Sinusoid object"""
    return Sinusoid(freq, amp, offset, func=np.sin)
