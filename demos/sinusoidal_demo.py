"""Demo for plot sinusoidal functions: sin, cos and sum of both"""

import thinkdsp
import matplotlib.pyplot as plt

if __name__ =='__main__':

    cos_sig = thinkdsp.cos_signal(freq=1, amp=1.0, offset=0)
    sin_sig = thinkdsp.sin_signal(freq=1, amp=1.0, offset=0)
    mix_sig = cos_sig + sin_sig

    wave_cos = cos_sig.make_wave(duration=1, start=0, framerate=11025)
    wave_sin = sin_sig.make_wave(duration=1, start=0, framerate=11025)
    mix = mix_sig.make_wave(duration=1, start=0, framerate=11025)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wave_cos.ts, wave_cos.ys, color='blue')
    ax.plot(wave_sin.ts, wave_sin.ys, color='orange')
    ax.plot(mix.ts, mix.ys, color='red')
    ax.set_title('Sin, cosine and mix signals')
    plt.show()
