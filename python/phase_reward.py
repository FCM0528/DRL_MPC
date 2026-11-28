"""Independent station quadrature and causal maxima for reward v3."""
import numpy as np


def phase_reward_terms(data,bounds,refs,decision_steps,gamma):
    x=data['x_m']
    previous=data['previous_x_m']
    error=np.abs(data['ey_m'])
    start,split,end=bounds
    edges=np.asarray([[start,split],[split,end]],dtype=float)
    crossed=np.maximum(0.,np.minimum(x[:,None],edges[:,1])-np.maximum(previous[:,None],edges[:,0]))
    fraction=crossed/np.diff(edges,axis=1).ravel()
    linear=fraction*error[:,None]/refs[0]
    square=fraction*(error[:,None]/refs[1])**2
    inside=np.column_stack(((x>=start)&(x<split),(x>=split)&(x<=end)))
    worst=np.max(np.cumsum(linear,axis=0),axis=1)
    peaks=np.maximum.accumulate(inside*error[:,None]/refs[2],axis=0)
    peak=np.max(peaks,axis=1)
    discount_scale=20.*decision_steps*gamma**(-(np.arange(len(x))//decision_steps))
    tracking=.25*linear.sum(axis=1)+.075*square.sum(axis=1)+.25*np.diff(worst,prepend=0.)
    beta=np.where(np.isfinite(data['measured_beta_rad']),data['measured_beta_rad'],data['sideslip_rad'])
    stability=np.maximum(0.,(np.abs(np.column_stack((data['body_ay_mps2'],beta)))/
                             [.85*9.80665,np.deg2rad(5)]-.8)/.2)
    peak_step=.15*np.diff(peaks.sum(axis=1),prepend=0.)+.20*np.diff(peak,prepend=0.)
    return dict(wave_tracking=discount_scale*tracking,wave_peak=discount_scale*peak_step,
                safety_addition=.25*(stability**2).sum(axis=1))
