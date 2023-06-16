#
# Contact Damped Oscillator
#
#

import autograd
import autograd.numpy as np
import torch as tc
import scipy.integrate
from integrators import contact as ci
from physis import defsys
solve_ivp = scipy.integrate.solve_ivp


def hamiltonian_fn(coords):
    q, p ,s = np.split(coords,3)
    H = p**2/2. + q**2/2. + s/10. # spring hamiltonian (linear oscillator)
    return H

def contact_mat(coords):
    p=coords[1]
    mat=np.array([[0,1,0],
                  [-1,0,-p],
                  [0,p,0]])            
    return mat

def k2_fn(coords):
    q, p,s = np.split(coords,3)
    k2= 2*s - q*p
    return k2

def kofmotion(coords):
    f=k2_fn(coords)/hamiltonian_fn(coords)
    return f

def kofmotiongrad(coords):
    df=autograd.grad(kofmotion)(coords)
    return df

def gradH(coords):
    dh=autograd.grad(hamiltonian_fn)(coords)
    return dh

def dynamics_fn(t, coords):
    # computation of the gradient of the Hamiltonian
    hams = hamiltonian_fn(coords)
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    #Setting the Hamiltonian equations into a vector
    if np.size(coords)>3:
        # if we have a set of coordinates
        dqdt, dpdt , dsdt = np.split(np.einsum('ijk,jk->ik',contact_mat(coords),dcoords),3)
    if np.size(coords)==3:
        dqdt, dpdt, dsdt = np.split(np.matmul(contact_mat(coords),dcoords),3)
    dsdt=dsdt-hams 
    # probably to fix using shape sqeeze stack split 
        
    S = np.concatenate([dqdt, dpdt, dsdt], axis=-1)
    return S

def get_trajectory(hamint = False, t_span=[0,6], timescale=5, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(3)*2-1
    if radius is None:
        radius = np.random.rand()*5 + 0.5 # sample a range of radii
    y0[:1] = y0[:1] * radius ## set the appropriate radius
    y0[2]  = y0[2] * 5
    if hamint:
        #Hamilonian Integrators
        osc=defsys()
        q0, p0, s0 = np.split(y0,3)
        sol, sols, _ =  ci.integrate(ci.step1q, osc, t_eval, p0, q0, s0)
        sols=sols.reshape(len(sols),1)
        sol[:,[1,0]]=sol[:,[0,1]]
        sol=sol.squeeze()
        sol=np.append(sol,sols,axis=-1)
        q ,p, s  = sol[:,0], sol[:,1], sol[:,2]
        dydt = [dynamics_fn(None, y) for y in sol]
    else:    
        spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        q, p , s= spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2]
        dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt, dsdt= np.split(dydt,3)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    s += np.random.randn(*s.shape)*noise_std
    return q, p, s, dqdt, dpdt, dsdt, t_eval

def get_dataset(hamint=False, seed=0, samples=25, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xl, dxl = [], []
    for l in range(samples):
        x, y, z, dx, dy, dz, t = get_trajectory(**kwargs)
        xl.append( np.stack( [x, y, z]).T )
        dxl.append( np.stack( [dx, dy, dz]).T )
        
    data['x'] = np.concatenate(xl)
    data['dx'] = np.concatenate(dxl).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2 , ymin=-1.2, ymax=1.2, zmin=-1.2,zmax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field:
    # here a grid of size in the phase space
    # b are the x coordinates of the set
    # a are the y coordinates of the set of the points
    c, b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize), np.linspace(zmin, zmax, gridsize))
    # ys is the grid with 2 x gridsize x gridsize entries
    yl = np.stack([c.flatten(), b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in yl.T]
    # producing the Hamiltonian vector fields grid
    # 
    dydt = np.stack(dydt).T

    field['x'] = yl.T
    field['dx'] = dydt.T
    return field