"""Demo for plot sinusoidal functions: sin, cos and sum of both
shows a segment of 3 times of period"""

import thinkdsp
import matplotlib.pyplot as plt

if __name__ =='__main__':

    cos_sig = thinkdsp.cos_signal(freq=440, amp=1.0, offset=0)
    sin_sig = thinkdsp.sin_signal(freq=880, amp=0.5, offset=0)
    mix_sig = cos_sig + sin_sig

    period = mix_sig.period

    wave_cos = cos_sig.make_wave(duration=0.5, start=0, framerate=11025).segment(start=0, duration=3*period)
    wave_sin = sin_sig.make_wave(duration=0.5, start=0, framerate=11025).segment(start=0, duration=3*period)
    mix = mix_sig.make_wave(duration=0.5, start=0, framerate=11025).segment(start=0, duration=3*period)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wave_cos.ts, wave_cos.ys, color='blue')
    ax.plot(wave_sin.ts, wave_sin.ys, color='orange')
    ax.plot(mix.ts, mix.ys, color='red')
    ax.set_title('Sin, cosine and mix signals')
    plt.show()
