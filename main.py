"""File to test DSP modules"""

import thinkdsp
import matplotlib.pyplot as plt

if __name__ =='__main__':

    cos_sig = thinkdsp.cos_signal(freq=1, amp=1.0, offset=0)
    wave = cos_sig.make_wave(duration=3, start=0, framerate=11025)
    wave.plot()
    plt.show()
