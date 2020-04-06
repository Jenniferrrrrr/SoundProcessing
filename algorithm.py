import soundfile as sf
import numpy as np
from scipy import signal
import sys
#smooth the given input by taking the moving average across n samples
def moving_average_smoothing(sound, output,n):
    y, syx = sf.read(sound)
    #moving average is actually a special case of convolution
    out = np.convolve(y, np.ones((n,))/n, mode='valid') 
    sf.write (output, out, srx)

def next_power_of_2(x): #used idea from class 
    if x == 0:
        return 1 
    return 2**(x - 1).bit_length()

#deconvolution using inverse filter, which is sensitive to noise
def deconvolution_inverse (out,impulse,output): 
    y, syx = sf.read(out)
    x, srx = sf.read(impulse)

    N = next_power_of_2(y.shape[0])   
    #get the inverse directly
    rebuild = np.fft.irfft(np.fft.fft (y, N) / np.fft.fft (x, N))

    sig_len = y.shape[0] - x.shape[0]
    rebuild = rebuild[1:sig_len]

    sf.write (output, rebuild, srx)
#helper method that calculates the convolution matrix given vector v and dimension n
def convolution_matrix(v, n):
    N = len(v) + 2*n - 2
    padding = np.concatenate([np.zeros(n-1), v[:], np.zeros(n-1)])
    matrix = np.zeros((len(v)+n-1, n))
    for i in xrange(n):
        matrix[:,i] = padding[n-i-1:N-i]
    return matrix.transpose()

#deconvolution using least squares, a more general way than the weiner used in class
def deconvolution_least_squares(out,impulse,output):
    y, syx = sf.read(out)
    h, srx = sf.read(impulse)
    H = convolution_matrix(h,y.shape[0]) #generate the convolution matrix from the impulse
    H = H(1:y.shape[0], :)
    lam = 0.1 #lambda value, which can be adjusted
    #call the least square formula
    rebuild = (np.matmul(np.transpose(H),H) + lam * np.eye(y.shape[0])) / (np.matmul(np.transpose(H), y))
    sf.write (output, rebuild, srx)

#here are two possible filtering processes:

def l_filt(low, high, order, sr, data): #does filtering with lfilter
    b,a = signal.butter(order, [low, high], fs=sr, btype='band') 
    return signal.lfilter(b,a,data) 

#does filtering with second-order sections, which gives good coefficients a&b when order is large
def sos_filt(low, high, order, sr, data): 
    b,a = signal.butter(order, [low, high], fs=sr, btype='band', output='sos') 
    return signal.sosfilt(b,a,data)

#heterodyne sound1 with sound2 and output the filtered result
def heterodyning (sound1, sound2, output, low, high): #the range of values we need to filter
	x, srx = sf.read(sound1)
	h, srh = sf.read(sound2)
	
	if srx != srh:
		sys.exit('sr must be the same in both files')
	if x.shape[0] > h.shape[0]:
		N = next_pow_of_2(x.shape[0])
	else:
		N = next_power_of_2(h.shape[0])
		
	y1 = np.fft.rfft(x, N)
	y2 = np.fft.rfft(h, N)
    result = np.fft.irfft(np.maximum(y1 + y2,y1 - y2)) # get the sum and difference and make it into one plot 
    #specify some order to apply the filter with & scale the range
    order = 5 
    low = low/(0.5*srx)
    high = high/(0.5*srx)
    #call one of the two filtering process defined below
    result = l_filt(low, high, order, srx, result) #can be switched to sos_filt
	sf.write(output, result, srx)

