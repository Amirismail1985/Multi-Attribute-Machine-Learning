"""Seismic conditioning for 2D post-stack lines stored as arrays of shape (n_samples, n_traces).

Order used in the paper: despike -> band-pass -> spectral balancing -> edge-preserving dip-steered median filter.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
DT=0.004
def bandpass(d,f1=6,f2=60,order=4):
    sos=butter(order,[f1,f2],btype='band',fs=1/DT,output='sos'); return sosfiltfilt(sos,d,axis=0).astype(np.float32)
def despike(d,k=6):
    m=np.median(np.abs(d)); return np.clip(d,-k*m*3,k*m*3)
def spectral_balance(d,f1=6,f2=60,win=125,smooth=5):
    """time-variant spectral whitening inside the band, with overlapping windows (amplitude-preserving per window RMS)"""
    ns,nt=d.shape; out=np.zeros_like(d); wsum=np.zeros(ns)
    step=win//2; f=np.fft.rfftfreq(win,DT); taper=np.hanning(win)
    band=np.clip(np.minimum((f-f1+3)/3,(f2+5-f)/5),0,1)
    for s in range(0,ns-win+1,step):
        seg=d[s:s+win]*taper[:,None]; S=np.fft.rfft(seg,axis=0)
        A=np.abs(S).mean(1); A=np.convolve(A,np.ones(smooth)/smooth,'same')+1e-6
        Sw=S/A[:,None]*band[:,None]; segw=np.fft.irfft(Sw,n=win,axis=0)
        segw*=np.sqrt((seg**2).mean())/ (np.sqrt((segw**2).mean())+1e-9)
        out[s:s+win]+=segw*taper[:,None]; wsum[s:s+win]+=taper**2
    return (out/np.maximum(wsum,1e-3)[:,None]).astype(np.float32)
def structure_tensor(d,sg=1.0,si=(6,3)):
    gt=np.gradient(gaussian_filter(d,sg),axis=0); gx=np.gradient(gaussian_filter(d,sg),axis=1)
    Ttt=gaussian_filter(gt*gt,si); Txx=gaussian_filter(gx*gx,si); Ttx=gaussian_filter(gt*gx,si)
    # dip in samples/trace : reflector normal (gt,gx) -> dip p = -gx/gt
    lam1=0.5*(Ttt+Txx+np.sqrt((Ttt-Txx)**2+4*Ttx**2)); lam2=0.5*(Ttt+Txx-np.sqrt((Ttt-Txx)**2+4*Ttx**2))
    p=-Ttx/(Ttt+1e-9)  # approx dip
    lin=(lam1-lam2)/(lam1+lam2+1e-9)   # linearity / planarity 0..1
    return np.clip(p,-3,3),lin
def sof(d,half=3,iters=2,edge=True):
    """edge-preserving dip-steered median filter: median along local dip over +/-half traces,
    blended with input where linearity is low (discontinuities such as chimneys/faults are kept)"""
    out=d.copy()
    for _ in range(iters):
        p,lin=structure_tensor(out)
        ns,nt=out.shape; T,X=np.mgrid[0:ns,0:nt].astype(np.float32); stack=[]
        for k in range(-half,half+1):
            stack.append(map_coordinates(out,[T+k*p,np.clip(X+k,0,nt-1)],order=1,mode='nearest'))
        med=np.median(np.array(stack),0)
        w=np.clip((lin-0.3)/0.5,0,1) if edge else 1.0
        out=(w*med+(1-w)*out).astype(np.float32)
    return out
def semblance(d,half=2,tw=5):
    """eigen-free semblance along local dip (coherence); variance = 1-semblance"""
    p,_=structure_tensor(d); ns,nt=d.shape; T,X=np.mgrid[0:ns,0:nt].astype(np.float32)
    S=[map_coordinates(d,[T+k*p,np.clip(X+k,0,nt-1)],order=1,mode='nearest') for k in range(-half,half+1)]
    S=np.array(S); num=uniform_filter(S.sum(0)**2,(tw,1)); den=uniform_filter((S**2).sum(0),(tw,1))*S.shape[0]
    return (num/(den+1e-9)).astype(np.float32)


def condition(d, f1=6, f2=60):
    """Full conditioning chain of the paper. Returns (band-passed data, conditioned data).
    Frequency/amplitude attributes use the band-passed data (before spectral balancing);
    geometric attributes use the conditioned data."""
    d = despike(np.asarray(d, np.float32))
    bp = bandpass(d, f1, f2)
    bal = spectral_balance(bp, f1, f2)
    return bp, sof(bal)
