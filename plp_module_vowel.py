import numpy.matlib
from sidekit.frontend.features import *
from sklearn.cross_validation import train_test_split
import numpy.matlib
import os
import scipy.io.wavfile as wav
import numpy as np
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle



PARAM_TYPE = numpy.float32


# GLOBAL VARIABLES
order_val = 13
overlap = 0
window_length = 360


####################################

# This module contains both PLP-AP and PLP-SVM sub modules

####################################



# library functions used

def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.

    :param f: frequency to convert

    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z


def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.

    :param m: ndarray of frequencies to convert in Hz.

    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10 ** (z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f


def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    # print(f)
    # print(6. * numpy.arcsinh(f / 600.))
    # import time
    # time.sleep(10)
    return 6. * numpy.arcsinh(f / 600.)


def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * numpy.sinh(z / 6.)


def power_spectrum(input_sig,
                   fs=8000,
                   win_time=0.025,
                   shift=0.01,
                   prefac=0.97):
    """
    Compute the power spectrum of the signal.
    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :param prefac:
    :return:
    """
    # window_length = int(round(win_time * fs))
    global window_length
    global overlap
    # window_length = 360
    #overlap = window_length - int(shift * fs)
    # overlap = 0
    # print(window_length - overlap)
    framed = framing(input_sig, window_length, win_shift=window_length - overlap).copy()
    # print(framed)
    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)

    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=PARAM_TYPE)
    log_energy = numpy.log((framed ** 2).sum(axis=1))
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, n_fft, axis=-1)
        spec[start:stop, :] = mag.real ** 2 + mag.imag ** 2
        start = stop
        stop = min(stop + dec, l)

    return spec, log_energy



def fft2barkmx(n_fft, fs, nfilts=0, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(n_fft,fs) * abs(fft(xincols, n_fft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param n_fft: the source FFT size at sampling rate fs
    :param fs: sampling rate
    :param nfilts: number of output bands required
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """
    # print(n_fft)
    maxfreq = min(maxfreq, fs / 2.)

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if nfilts == 0:
        nfilts = numpy.ceil(nyqbark) + 1

    wts = numpy.zeros((nfilts, n_fft))

    # bark per filt
    step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(numpy.arange(n_fft / 2 + 1) * fs / n_fft)

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = (binbarks - f_bark_mid - 0.5)
        hif = (binbarks - f_bark_mid + 0.5)
        wts[i, :n_fft // 2 + 1] = 10 ** (numpy.minimum(numpy.zeros_like(hif), numpy.minimum(hif, -2.5 * lof) / width))


    return wts


def fft2melmx(n_fft,
              fs=8000,
              nfilts=0,
              width=1.,
              minfreq=0,
              maxfreq=4000,
              htkmel=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(n_fft,fs)*abs(fft(xincols,n_fft));
    minfreq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfreq is frequency in Hz of upper edge; default fs/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.

    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

    :param n_fft:
    :param fs:
    :param nfilts:
    :param width:
    :param minfreq:
    :param maxfreq:
    :param htkmel:
    :param constamp:
    :return:
    """
    maxfreq = min(maxfreq, fs / 2.)

    if nfilts == 0:
        nfilts = numpy.ceil(hz2mel(maxfreq, htkmel) / 2.)

    wts = numpy.zeros((nfilts, n_fft))

    # Center freqs of each FFT bin
    fftfrqs = numpy.arange(n_fft / 2 + 1) / n_fft * fs

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    binfrqs = mel2hz(minmel + numpy.arange(nfilts + 2) / (nfilts + 1) * (maxmel - minmel), htkmel)

    for i in range(nfilts):
        _fs = binfrqs[i + numpy.arange(3, dtype=int)]
        # scale by width
        _fs = _fs[1] + width * (_fs - _fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - _fs[0]) / (_fs[1] - _fs[0])
        hislope = (_fs[2] - fftfrqs) / (_fs[2] - _fs[1])

        wts[i, 1 + numpy.arange(n_fft // 2 + 1)] = numpy.maximum(numpy.zeros_like(loslope),
                                                                 numpy.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = numpy.dot(numpy.diag(2. / (binfrqs[2 + numpy.arange(nfilts)] - binfrqs[numpy.arange(nfilts)])), wts)

    # Make sure 2nd half of FFT is zero
    wts[:, n_fft // 2 + 1: n_fft] = 0

    return wts, binfrqs


def audspec(power_spectrum,
            fs=16000,
            nfilts=None,
            fbtype='bark',
            minfreq=0,
            maxfreq=8000,
            sumpower=True,
            bwidth=1.):
    """

    :param power_spectrum:
    :param fs:
    :param nfilts:
    :param fbtype:
    :param minfreq:
    :param maxfreq:
    :param sumpower:
    :param bwidth:
    :return:
    """
    if nfilts is None:
        nfilts = int(numpy.ceil(hz2bark(fs / 2)) + 1)

    if not fs == 16000:
        maxfreq = min(fs / 2, maxfreq)

    nframes, nfreqs = power_spectrum.shape
    n_fft = (nfreqs - 1) * 2

    if fbtype == 'bark':
        wts = fft2barkmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'mel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'htkmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, True)
    elif fbtype == 'fcmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, False)
    else:
        print('fbtype {} not recognized'.format(fbtype))


    import time


    np.set_printoptions(threshold=np.sys.maxsize)
    # print(type(wts))
    # print(nfreqs)
    # print(wts[0])
    wts = wts[:, :nfreqs]
    # print(wts[0])

    # print(power_spectrum[0])
    # time.sleep(10)

    if sumpower:
        audio_spectrum = power_spectrum.dot(wts.T)
    else:
        audio_spectrum = numpy.dot(numpy.sqrt(power_spectrum), wts.T) ** 2

    return audio_spectrum, wts


def postaud(x, fmax, fbtype='bark', broaden=0):
    """
    do loudness equalization and cube root compression

    :param x:
    :param fmax:
    :param fbtype:
    :param broaden:
    :return:
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    nfpts = nbands + 2 * broaden

    if fbtype == 'bark':
        bandcfhz = bark2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2mel(fmax, 1), num=nfpts), 1)
    else:
        print('unknown fbtype {}'.format(fbtype))

    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz ** 2
    ftmp = fsq + 1.6e5
    eql = ((fsq / ftmp) ** 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = numpy.matlib.repmat(eql.T, nframes, 1) * x

    # cube root compress
    z = z ** .33

    # replicate first and last band (because they are unreliable as calculated)
    if broaden == 1:
        y = z[:, numpy.hstack((0, numpy.arange(nbands), nbands - 1))]
    else:
        y = z[:, numpy.hstack((1, numpy.arange(1, nbands - 1), nbands - 2))]

    return y, eql


def dolpc(x, model_order=8):
    """
    compute autoregressive model from spectral magnitude samples

    :param x:
    :param model_order:
    :return:
    """
    nframes, nbands = x.shape

    r = numpy.real(numpy.fft.ifft(numpy.hstack((x, x[:, numpy.arange(nbands - 2, 0, -1)]))))

    # First half only
    r = r[:, :nbands]

    # Find LPC coeffs by Levinson-Durbin recursion
    y_lpc = numpy.ones((r.shape[0], model_order + 1))

    for ff in range(r.shape[0]):
        y_lpc[ff, 1:], e, _ = levinson(r[ff, :-1].T, order=model_order, allow_singularity=True)
        # Normalize each poly by gain
        y_lpc[ff, :] /= e

    return y_lpc


def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    nout is number of cepstra to produce, defaults to size(lpcas,1)
    2003-04-11 dpwe@ee.columbia.edu

    :param a:
    :param nout:
    :return:
    """
    ncol, nin = a.shape

    order = nin - 1

    if nout is None:
        nout = order + 1

    c = numpy.zeros((ncol, nout))

    # First cep is log(Error) from Durbin
    c[:, 0] = -numpy.log(a[:, 0])

    # Renormalize lpc A coeffs
    a /= numpy.tile(a[:, 0][:, None], (1, nin))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum += (n - m) * a[:, m] * c[:, n - m]
        c[:, n] = -(a[:, n] + sum / n)

    return c


def lifter(x, lift=0.6, invs=False):
    """
    Apply lifter to matrix of cepstra (one per column)
    lift = exponent of x i^n liftering
    or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == 1 (default 0), undo the liftering.

    :param x:
    :param lift:
    :param invs:
    :return:
    """
    nfrm, ncep = x.shape

    if lift == 0:
        y = x
    else:
        if lift > 0:
            if lift > 10:
                print('Unlikely lift exponent of {} did you mean -ve?'.format(lift))
            liftwts = numpy.hstack((1, numpy.arange(1, ncep) ** lift))

        elif lift < 0:
            # Hack to support HTK liftering
            L = float(-lift)
            if (L != numpy.round(L)):
                print('HTK liftering value {} must be integer'.format(L))

            liftwts = numpy.hstack((1, 1 + L / 2 * numpy.sin(numpy.arange(1, ncep) * numpy.pi / L)))

        if invs:
            liftwts = 1 / liftwts

        y = x.dot(numpy.diag(liftwts))

    return y


def plp(input_sig,
        nwin=0.025,
        fs=8000,
        plp_order=order_val,
        shift=0.01,
        get_spec=False,
        get_mspec=False,
        prefac=0.97,
        rasta=True):
    """
    output is matrix of features, row = feature, col = frame

    % fs is sampling rate of samples, defaults to 8000
    % dorasta defaults to 1; if 0, just calculate PLP
    % modelorder is order of PLP model, defaults to 8.  0 -> no PLP

    :param input_sig:
    :param fs: sampling rate of samples default is 8000
    :param rasta: default is True, if False, juste compute PLP
    :param model_order: order of the PLP model, default is 8, 0 means no PLP

    :return: matrix of features, row = features, column are frames
    """
    plp_order -= 1

    # first compute power spectrum
    powspec, log_energy = power_spectrum(input_sig, fs, nwin, shift, prefac)

    # next group to critical bands
    audio_spectrum = audspec(powspec, fs)[0]
    nbands = audio_spectrum.shape[0]

    if rasta:
        # put in log domain
        nl_aspectrum = numpy.log(audio_spectrum)

        #  next do rasta filtering
        ras_nl_aspectrum = rasta_filt(nl_aspectrum)

        # do inverse log
        audio_spectrum = numpy.exp(ras_nl_aspectrum)

    # do final auditory compressions
    post_spectrum = postaud(audio_spectrum, fs / 2.)[0]



    # LPC analysis
    lpcas = dolpc(post_spectrum, plp_order)

    # convert lpc to cepstra
    cepstra = lpc2cep(lpcas, plp_order + 1)

        # .. or to spectra
        # spectra, F, M = lpc2spec(lpcas, nbands)


    cepstra = lifter(cepstra, 0.6)

    lst = list()
    lst.append(cepstra)
    lst.append(log_energy)
    if get_spec:
        lst.append(powspec)
    else:
        lst.append(None)
        del powspec
    if get_mspec:
        lst.append(post_spectrum)
    else:
        lst.append(None)
        del post_spectrum

    return lst


def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context,) + (sig.ndim - 1) * ((0, 0),)
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()




def levinson(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.

    Find the coefficients of a length(r)-1 order autoregressive linear process

    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)

    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values


        >>> import numpy; from spectrum import LEVINSON
        >>> T = numpy.array([3., -2+0.5j, .7-1j])
        >>> a, e, k = LEVINSON(T)

    """
    # from numpy import isrealobj
    T0 = numpy.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = numpy.isrealobj(r)
    if realdata is True:
        A = numpy.zeros(M, dtype=float)
        ref = numpy.zeros(M, dtype=float)
    else:
        A = numpy.zeros(M, dtype=complex)
        ref = numpy.zeros(M, dtype=complex)

    P = T0

    for k in range(M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k - j - 1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp ** 2.)
        else:
            P = P * (1. - (temp.real ** 2 + temp.imag ** 2))

        if (P <= 0).any() and allow_singularity == False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp  # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k + 1) // 2
        if realdata is True:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp * save
        else:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref


# end of library functions


############################################


def readWavFile(wav):
    inputWav = os.path.abspath(wav)

    return inputWav

#reading the .wav file (signal file) and extract the information we need
def initialize(inputWav):
    rate , signal  = wav.read(readWavFile(inputWav)) # returns a wave_read object , rate: sampling frequency
    #print('signal', signal)
    #normalization
    max_sig = max(signal)
    for idx in range(len(signal)) :
        signal[idx] = signal[idx] * 5000 / max_sig
    return signal ,  rate

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def readTxtFile(wav):
    signal = []
    final_signal = []
    with open(wav, "r") as f1:
        buffer = f1.readlines()
        #print(buffer)
        for value in buffer:
            if isfloat(value) == True:
                signal.append(float(value))

    # normalisation
    max_sig = max(signal)
    for idx in range(len(signal)) :
        signal[idx] = signal[idx] * 5000 / max_sig

    # limiting the file_data to 14,400 = 360 * 40 frames for Experiment 2
    average_val = sum(signal) / len(signal)
    if len(signal) < 14400:
        while(len(signal)) < 14400:
            signal.append(average_val)

        final_signal = signal

    if len(signal) > 14400:
        mid_index = len(signal)/ 2
        end_index = 14400/2 + mid_index
        start_index = mid_index - 14400/2
        for index in range(start_index, end_index):
            final_signal.append(signal[index])

    while len(final_signal) > 14400:
        final_signal.pop()

    return numpy.asarray(final_signal)

def PLP(folder_name, experiment_path):
    # folder = "test/txtFilesTestData"   # for text files test
    #folder = "test/wavFilesTestData"      # for wav files test
    folder = folder_name
    lstPlpFeatures = []
    l = os.listdir(experiment_path + folder)
    # print(l)
    print('folder_name', folder_name)
    for x in sorted(l):
        # print(x)
        wav = experiment_path + folder+'/'+ x
        inputWav = readWavFile(wav)
        #signal,rate = initialize(wav)       #for wav file
        signal = readTxtFile(wav)           #for txt file
        #print(type(signal))
        #returns PLP coefficients for every frame
        plp_features = plp(signal, rasta=True)
        #print(plp_features[0].ravel().tolist())
        #item = meanFeatures(plp_features[0])
        #lstPlpFeatures.append(item.tolist())
        lstPlpFeatures.append(plp_features[0].ravel().tolist())             # for experiment 2, without mean features
    return lstPlpFeatures


#compute the mean features for one .wav file (take the features for every frame and make a mean for the sample)
def meanFeatures(plp_features):
    #make a numpy array with length the number of plp features
    mean_features=np.zeros(len(plp_features[0]))
    #for one input take the sum of all frames in a specific feature and divide them with the number of frames
    for x in range(len(plp_features)):
        cnt = 0
        for y in range(len(plp_features[x])):
            cnt +=1
            mean_features[y]+=plp_features[x][y]
    mean_features = (mean_features / len(plp_features))
    return mean_features





def ap(mean_features_list, speaker, speaker_cnt):
    global over_all_count
    global  over_all_accuracy
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1], [-1, -1]]
    X =numpy.asarray(mean_features_list)
    print('order_val', order_val)
    # print( 'mean features list',X)
    dict_points_to_indices = {}
    for point,indice in zip(X, range(len(X))):
        #print(point,indice)
        dict_points_to_indices[point.tostring()] = indice+1


    # Compute Affinity Propagation
    af = AffinityPropagation().fit(X)
    # print 'affinity propagation ', af
    cluster_centers_indices = af.cluster_centers_indices_
    print ('cluster centers indices', cluster_centers_indices)
    #print( 'cluster_centers_', af.cluster_centers_)
    print('labels_', af.labels_)
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    print ('n clusters length', n_clusters_)



    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')



    plt.rc('axes', titlesize=25)

    fig,ax = plt.subplots()
    plt.ylabel('y - axis : negative squared distance', fontsize=22)
    plt.xlabel('x - axis : negative squared distance', fontsize=22)
    plt.tick_params(labelsize=20)

    cnt = 0
    var = 0
    acc_0 = 0
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0


    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        #print('class members', class_members)
        #print('X[class members]', X[class_members])
        cluster_center = X[cluster_centers_indices[k]]
        #print(X[class_members, 0], X[class_members, 1])
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markerfacecolor=col, markeredgecolor='k',
                 markersize=14)

        for x in X[class_members]:
            cnt = cnt+1
            plt.plot([cluster_center[0], x[0]],
                     [cluster_center[1], x[1]], col)
            #strm = str(cnt)
            val = cluster_centers_indices
            strm = str(dict_points_to_indices[x.tostring()]) +"," + str(k)
            #print(str(cnt) +"," + str(k))
            ax.annotate(strm, xy=(x[0], x[1]))
            # if cnt - k * 5 > 0 :
            #     var +=1

            #accuracy calculation

            actual_digit = (cnt-1)/10
            cluster_center_for_it = labels[cnt-1]
            actual_digit_for_cluster = (cluster_centers_indices[cluster_center_for_it])/10
            if(actual_digit == actual_digit_for_cluster) :
                var = var+1

                if(actual_digit_for_cluster == 0):
                    acc_0 += 1
                elif (actual_digit_for_cluster == 1):
                    acc_1 += 1
                elif (actual_digit_for_cluster == 2):
                    acc_2 += 1
                elif (actual_digit_for_cluster == 3):
                    acc_3 += 1
                elif (actual_digit_for_cluster == 4):
                    acc_4 += 1



    print("accuracy ",var* 100/cnt)
    over_all_accuracy += var* 100/cnt
    over_all_count += 1
    print( "individual accuracy for speaker : ",speaker, ', Individual Accuracy for vowels : ', acc_0 *100/10, acc_1*100/10, acc_2*100/10, acc_3*100/10, acc_4*100/10)

    plt.title('Estimated number of clusters: % d' % n_clusters_)
    fig.set_size_inches(18.5, 10.5, forward=True)

    # fig.savefig('/home/akk/Documents/speech processing BTP/Experiments/Experiment1/plots/' + str(order_val) +'/' + speaker +'.png')   # save the figure to file
    # plt.close(fig)

    #plt.show()





over_all_accuracy = 0
over_all_count = 0
accuracy_list = list()


def svm_function(features_list):

    vowel_indication_list = list()

    for i in range(0,5):
        vowel_indication_list =  vowel_indication_list + ([i] * 10)


    X = features_list

    y = numpy.array(vowel_indication_list)


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.20)

    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    global over_all_accuracy
    global over_all_count
    print('accuracy' ,accuracy)
    accuracy_list.append(accuracy)
    over_all_accuracy = over_all_accuracy + accuracy
    over_all_count += 1


def Experiment_PLP_SVM():


    print("#############################")
    print("        PLP SVM MODULE       ")
    print("#############################")
    directory_list = list()
    # for root, dirs, files in os.walk("/home/akk/Documents/speech processing BTP/Experiments/Experiment1/data", topdown=False):
    #     for name in sorted(dirs):
    #         directory_list.append(name)



    dir_path = os.path.abspath(os.getcwd())
    # print(dir_path + "/Experiments/Experiment1/data")

    for root, dirs, files in os.walk(dir_path + "/Experiments/Experiment1/data", topdown=False):
        for name in sorted(dirs):
            directory_list.append(name)



    # print directory_list
    # import time
    # time.sleep(10)
    path = 'Experiments/Experiment1/data/'


    # os.mkdir('/home/akk/Documents/speech processing BTP/Experiments/Experiment1/plots/' + str(order_val))
    speaker_cnt = 0
    for speaker in sorted(directory_list):
        speaker_cnt = speaker_cnt + 1
        features_list = PLP(speaker, path)

        # print(numpy.array(mean_features_list))
        svm_function(features_list)


def Experiment_PLP_AP():

    print("#############################")
    print("        PLP AP MODULE       ")
    print("#############################")


    dir_path = os.path.abspath(os.getcwd())

    directory_list = list()
    for root, dirs, files in os.walk(dir_path + "/Experiments/Experiment1/data", topdown=False):
        for name in sorted(dirs):
            directory_list.append(name)

    #print directory_list
    path = 'Experiments/Experiment1/data/'


    # os.mkdir('/home/akk/Documents/speech processing BTP/Experiments/Experiment1/plots/' + str(order_val))
    speaker_cnt = 0
    for speaker in sorted(directory_list):
        speaker_cnt = speaker_cnt + 1
        mean_features_list = PLP(speaker, path)
        ap(mean_features_list, speaker, speaker_cnt)



def calculateAccuracyVariance():
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    variance_accuracy = 0
    for accuracy in accuracy_list :
        variance_accuracy = variance_accuracy + (accuracy-mean_accuracy) * (accuracy-mean_accuracy)

    print('accuracy_list', accuracy_list)
    print('variance_accuracy', variance_accuracy)


def main():

    import warnings
    warnings.filterwarnings("ignore")

    global order_val
    global window_length
    input1 = raw_input("Enter the value for PLP order (default = 13) : ")
    if input1:
        order_val = input1

    input2 = raw_input("Enter the value for Window length (default = 360) : ")
    if input2:
        window_length = input2


    input3 = raw_input("Enter 1 for PLP-AP Module else any other number for PLP-SVM : ")
    if int(input3) == 1:
        Experiment_PLP_AP()
    else:
        Experiment_PLP_SVM()
        calculateAccuracyVariance()



    print("-----------")
    print('over all accuracy', over_all_accuracy/ over_all_count)
    print("-----------")


main()
