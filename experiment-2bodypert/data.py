import autograd
import autograd.numpy as np
import torch as tc
import scipy.integrate
from integrators import contact as ci 
from numpy.linalg import norm
from physys import TimePerturbedKepler
from utils import to_pickle, from_pickle
#from alive_progress import alive_bar
solve_ivp = scipy.integrate.solve_ivp
alpha=0.1
theta=2*np.pi
gamma=-1.
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def hamiltonian_fn(coords):
    t, q1,q2, p1, p2,s = np.split(coords,6)
    q=np.array([q1,q2]).squeeze()
    p=np.array([p1,p2]).squeeze()
    H = np.dot(p,p)/(2)  + (gamma)/(np.sqrt(np.dot(q,q))) +alpha*s*np.sin(theta*t) 
    return H

def contact_mat(coords):
    p=coords[2:4].squeeze()
    mat=np.array([[0 ,0 ,1   ,0   ,0],
                  [0 ,0 ,0   ,1   ,0],
                  [-1,0 ,0   ,0   ,-p[0]],
                  [0 ,-1,0   ,0   ,-p[1]],
                  [0 ,0 ,p[0],p[1],0]])
    return mat

def init_ecc(e):
    v=np.array([0.0, np.sqrt((1+e)/(1-e)),1.0 - e, 0.0])
    return v

def gradH(x):
    dh=autograd.grad(hamiltonian_fn)(x)
    return dh

def dynamics_fn(t,coords):
    # computation of the gradient of the Hamiltonian
    x=np.append([t],coords)
    hams = hamiltonian_fn(x)
    dcoords = autograd.grad(hamiltonian_fn)(x)[1:]
    #Setting the Hamiltonian equations into a vector
    dq1dt, dq2dt, dp1dt, dp2dt, dsdt = np.split(np.einsum("ij,j->i",contact_mat(coords),dcoords),5)
    dsdt=dsdt-hams 
    # probably to fix using shape sqeeze stack split 
    S = np.concatenate([dq1dt, dq2dt, dp1dt, dp2dt, dsdt], axis=-1)
    return S

def get_trajectory(hamint=False, ecc=0.4,t_span=None, timescale=20, radius=None, y0=None, noise_std=0.01, **kwargs):
    # get initial state
    if radius is None:
        radius= np.random.rand()*0.5+0.75 #(2.*np.random.rand()-1.)*5
        #radius = 1.
    if y0 is None:
        y0 = init_ecc(ecc)
        y0=np.append(y0,(np.random.rand()*2. -1.)*10.)
    y0[:1]*=radius
    
    if t_span is None:
        t_inf = (np.random.rand()*2.-1.)*10.
        delta = 150.*np.random.rand()+10.0
        t_span = [t_inf, t_inf+delta]
        del t_inf, delta
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if hamint:
        sys=TimePerturbedKepler(alpha,theta,gamma)
        q01, q02, p01, p02,s0= np.split(y0,5)
        p0=np.array([p01,p02]).squeeze()
        q0=np.array([q01,q02]).squeeze()
        sol, sols, _ = ci.integrate(ci.step1l,
                                   sys,
                                   t_eval,
                                   p0,q0,s0)
        sols=sols.reshape(len(sols),1)
        #control to see how sol is in this secti
        sol[:,[1,0],:]=sol[:,[0,1],:]
        psol=sol.reshape(np.shape(sol)[0],4)
        psol=np.append(psol,sols,axis=-1)
        q1, q2, p1, p2, s = psol[:,0], psol[:,1], psol[:,2], psol[:,3], psol[:,4]
        psol=np.append(t_eval.reshape(len(t_eval),1), psol, axis=-1)
        t = t_eval
    else:
        spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-8, **kwargs)
        q1,q2,p1, p2 , s , t= spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3], spring_ivp['y'][4], spring_ivp['t']
        psol = np.append(spring_ivp['t'].reshape(len(spring_ivp['t']),1),spring_ivp['y'].T,axis=-1)
        
    dydt=[dynamics_fn(t=vec[0],coords=vec[1:]) for vec in psol]   
    #
    #
    dydt = np.stack(dydt).T
    dq1dt, dq2dt, dp1dt, dp2dt, dsdt= np.split(dydt,5)
    # add noise
    q1 += np.random.randn(*q1.shape)*noise_std
    q2 += np.random.randn(*q2.shape)*noise_std
    p1 += np.random.randn(*p1.shape)*noise_std
    p2 += np.random.randn(*p2.shape)*noise_std
    s += np.random.randn(*s.shape)*noise_std
    return q1, q2, p1, p2, s, dq1dt, dq2dt, dp1dt, dp2dt, dsdt, t

def get_dataset(seed=0, samples=100, test_split=0.2, ecc=0.4, hamint=False, save_dir=None, **kwargs):
    data = {'meta': locals()}
    
    if hamint:
        path = '{}/{}-hamint-orbits-dataset.pkl'.format(save_dir, 'twobody')
    else:
        path = '{}/{}-orbits-dataset.pkl'.format(save_dir, 'twobody') 
        
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("No Data founded from {}".format(path))
        # randomly sample inputs
        np.random.seed(seed)
        xl, dxl = [], []
        #with alive_bar(samples) as bar:
        for l in range(samples):
            x1, x2, y1, y2, z, dx1,dx2,dy1,dy2, dz, t = get_trajectory(ecc=ecc, hamint=hamint, **kwargs)
            xl.append( np.stack( [t, x1,x2, y1,y2, z]).T )
            dxl.append( np.stack( [dx1,dx2,dy1,dy2, dz]).T )
            #bar()
        
        data['x'] = np.concatenate(xl)
        data['dx'] = np.concatenate(dxl).squeeze()

        # make a train/test split
        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'dx']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data
        to_pickle(data, path)
    return data
