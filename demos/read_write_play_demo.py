"""Demo for read, write and play WAV files"""

import thinkdsp

if __name__ =='__main__':

    # Play a wav file
    wave = thinkdsp.play_wave("sounds/sound.wav")

    #TODO: Read a wave file modifying, write and play