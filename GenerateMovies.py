#!/usr/bin/env python

# Random scale trainer by ytsysd@gmail.com
# This program generates a movie that randomly displays 
# and plays scale notes with a musical score. 

import numpy as np
from scipy.io import wavfile
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import matplotlib as mpl
import matplotlib.patches as patches
font = {'family': 'IPAPGothic'}
mpl.rc('font', **font)

WAV_RATE=8000 #8kHz causes mild tone! #WAV_RATE=44100
FPS=10


def get_wave_Ped(A=[0.25, 0.5], Hz=[440.0, 880.0], sec=[1.0, 3.0]):

    # Time and phase
    ts_root = np.linspace(0.0, sec[0]+sec[1], int(WAV_RATE * (sec[0]+sec[1])))
    ts_note = np.linspace(0.0, sec[1], int(WAV_RATE * sec[1]))
    def get_phase( freq, ts ):
        return 2.0 * np.pi * freq * ts

    # Waves
    def generate_wave( Amp, freq, ts ):
        ratio = 0.7
        ph1 = get_phase( freq, ts )
        ph2 = get_phase( 2.0*freq, ts )
        decay = 0.65
        w1 = np.exp(-decay*ts)*np.sin(ph1)  
        w2 = np.exp(-2.0*decay*ts)*np.sin(ph2)
        wave = Amp*(ratio*w1 + (1-ratio)*w2)
        return (wave * float(2 ** 15 - 1)).astype(np.int16)  #16bit

    root = generate_wave( A[0], Hz[0], ts_root )
    note = generate_wave( A[1], Hz[1], ts_note )

    note_idx0 = len(ts_root) - len(ts_note) - 1
    root[note_idx0:-1] = root[note_idx0:-1] + note

    return root


class Trainer:

    def __init__( s, nsound, ntry, offset=0, root_note=0, minor=False ):
        if minor is False:
            s.set_to_major()
        else:
            s.set_to_minor()
        s.train_data = s.get_train_data( nsound, ntry, offset )
        s.root_note = root_note
        print(s.train_data)

    def set_to_major(s):
        C4 = 220.0*(2.0**(3/12.0)) #C4 = 261.626
        C3 = C4/2.0
        s.R = C3 #root
        s.d_half = [2,2,1,2,2,2,1] #0,2,4,5,7,9,11,12

    def set_to_minor(s):
        A3 = 220.000
        s.R = A3 #root
        s.d_half = [2,1,2,2,1,2,2] #0,2,3,5,7,8,10,12

    def interval_to_half_index( s, math_interval ):
        half_index = 0
        if math_interval>0:
            for i in range(math_interval):
                half_index += s.d_half[i%7]
        elif math_interval<0:
            inv_d_half = [elm for elm in reversed(s.d_half)]
            for i in range(-math_interval):
                half_index -= inv_d_half[i%7]

        return half_index

    def getFreq( s, math_interval ):

        half_index = s.interval_to_half_index( math_interval )
        freq = s.R*2.0**(half_index/12.0)

        return freq

    def interval_octave( s, math_interval ):

        interval = (math_interval)%7 + 1
        octave = (math_interval)//7

        return (interval, octave)

    def random_sample( s, nsound, offset ):
        return rnd.sample(range(offset, nsound+offset),nsound)

    def get_train_data( s, nsound, ntry, offset ):
        counter = 0
        train_data = []
        cur_data = s.random_sample(nsound,offset)
        while len(train_data) < ntry:
            local_counter = counter%nsound
            if local_counter == 0:
                prev_data = cur_data
                while True:
                    cur_data = s.random_sample(nsound,offset)
                    if cur_data[0] != prev_data[-1]:
                        break

            train_data.append(cur_data[local_counter])

            counter += 1

        return train_data

    def get_wav_data_Ped( s ):
        for i, math_interval in enumerate(s.train_data):
            freq = s.getFreq(math_interval)
            root = s.getFreq(s.root_note)
            cur_wave = get_wave_Ped(A=[0.5, 0.7], 
                                    Hz=[root, freq], sec=[1.0, 3.0])
            if i==0:
                wave = cur_wave
            else: 
                wave = np.hstack((wave, cur_wave))

        return wave

            
    def get_anim_data( s ):

        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
        ax.tick_params(labelleft="off",left="off") # y軸の削除
        ax.set_xticklabels([])

        def clear_plot():
            ax.clear()
            ax.plot([1,1,-1,-1],[-1,1,-1,1],"o")

        def plot_score( math_interval = None ):
            dy = 0.13
            xlong  = [-0.3, 0.3 ]
            xshort = [-0.12, 0.12 ]

            for i in range(-2,3):
                ax.plot( xlong, dy*i*np.ones(2), 'k-' )

            if math_interval is None:
                return

            if math_interval <= 0:
                nlines = (2-math_interval)//2
                for i in range(nlines):
                    ax.plot( xshort, dy*(-3-i)*np.ones(2), 'k-' )
            elif math_interval >= 12:
                nlines = (math_interval-10)//2
                for i in range(nlines):
                    ax.plot( xshort, dy*(3+i)*np.ones(2), 'k-' )

            ynote = 0.5*dy*(math_interval-6)
            e = patches.Ellipse(xy=(np.mean(xshort), ynote), fc='k', ec='k',
                    width=1.0*dy, height=0.8*dy, angle=30)
            ax.add_patch(e)

            return ynote

        def plot( frame ):
            HL=["-Low", "Low", "Mid", "High"]
            Doremi = [u'ド', u'レ', u'ミ', u'ファ', u'ソ', u'ラ', u'シ']
            h=[-0.6, -0.6, -0.1, 0.5]
            interval_idx = frame//(4*FPS)
            math_interval = s.train_data[interval_idx]
            clear_plot()
            plot_score()
            frame_mod = frame%(4*FPS)
            color="green"
            if frame_mod >=FPS: 
                ynote = plot_score( math_interval )
                interval, octave = s.interval_octave( math_interval )
                ax.text(0.37, ynote, '%d' %(interval),
                        fontsize=32, ha='left', va='center', color=color)
                ax.text(-0.31, ynote, '%s ' %(Doremi[interval-1]),
                        fontsize=28, ha='right', va='center', color=color)
            fig.canvas.update()

        anim = animation.FuncAnimation(fig, plot,
                interval=1000/FPS,
                frames=FPS*4*len(s.train_data))

        return anim


def generate_movie( mode ):

    if ( mode is 'Cmaj_low' ):  
        tr = Trainer( nsound=8, offset=0, ntry=8*8 )
        output_file='Cmaj_low.mp4'

    elif ( mode is 'Cmaj_mid' ):  
        tr = Trainer( nsound=8, offset=7, ntry=8*8 )
        output_file='Cmaj_mid.mp4'

    elif ( mode is 'Cmaj_Cform' ): # 5th-string root
        tr = Trainer( nsound=17, offset=-4, ntry=17*5 )
        output_file='Cmaj_Cform.mp4'

    elif ( mode is 'Cmaj_Aform' ): # 5th-string root
        tr = Trainer( nsound=16, offset=-3, ntry=16*5 )
        output_file='Cmaj_Aform.mp4'

    elif ( mode is 'Cmaj_Gform' ): # 6th-string root
        tr = Trainer( nsound=17, offset=-2, ntry=17*5 )
        output_file='Cmaj_Gform.mp4'

    elif ( mode is 'Cmaj_Eform' ): # 6th-string root
        tr = Trainer( nsound=17, offset=-1, ntry=17*5 )
        output_file='Cmaj_Eform.mp4'

    elif ( mode is 'Cmaj_Dform' ): # 4th-string root
        tr = Trainer( nsound=17, offset=1, ntry=17*5, root_note=7 )
        output_file='Cmaj_Dform.mp4'

    elif( mode is 'test' ): 
        tr = Trainer( nsound=5, offset=0, ntry=5 )
        offset_math_root=0 
        output_file='test.mp4'

    else:
        print( "Unknown mode:" + mode )
        exit()

    wave = tr.get_wav_data_Ped()
    wavfile.write("temp.wav", WAV_RATE, wave)

    anim = tr.get_anim_data()
    anim.save("temp.mp4")

    subprocess.run(['ffmpeg', 
                    '-y', #Overwrite without asking
                    '-i', 'temp.wav', 
                    '-i', 'temp.mp4', 
                    '-c:v', 'libx264', output_file])

    subprocess.run(['rm', '-f', 'temp.wav'])
    subprocess.run(['rm', '-f', 'temp.mp4'])


generate_movie( 'Cmaj_low' )
generate_movie( 'Cmaj_mid' )
generate_movie( 'Cmaj_Cform' ) #C-form
generate_movie( 'Cmaj_Aform' ) #A-form
generate_movie( 'Cmaj_Gform' ) #G-form
generate_movie( 'Cmaj_Eform' ) #E-form
generate_movie( 'Cmaj_Dform' ) #D-form

#generate_movie( 'test' )
