import torch
import numpy as np

from nn_models import MLP
from utils import rk4
from integrators import contact as ci

#####################################
#           Autonomus HNN           #
# without dependences on the time   #
#####################################


class HNN(torch.nn.Module):
    # the x vector is decomposed into
    # x=[q1,...,qn,p1,..pn,s]
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim-1) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)
    
    def contact_tensorhd(self, x):
        # Canonical coordinates: 2 m + 1
        # (q1,...,qm,p1,...,pm,s)
        #this function define the matrix
        # 0  1  0
        # -1 0 -p
        # 0  p  0
        m=(x[0,:].size()[0])//2
        # here i can define m as the tensor of dimension and then define from that l.
        p=x[:,m:2*m] #cut the momenta
        l=p.size()[0] #number of entries
        mat0=torch.eye(m)
        mat=torch.zeros(l,2*m+1,2*m+1)
        for i in range(l):
            mat[i,m:2*m,:m]=-mat0
            mat[i,:m,m:2*m]= mat0
            mat[i,2*m,m:2*m] = p[i,:]
            mat[i,m:2*m,2*m] = - p[i,:]      
        return mat
    
    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, contact_field=True):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass

        horizontal_field = torch.zeros_like(x) # start out with both components set to 0
        vertical_field = torch.zeros_like(x)
        contmat=self.contact_tensorhd(x)
        m=x.shape[1]
        if self.field_type != 'horizontal':
            #This part computes the Hamiltonian vector field of the related Hamiltonian.
            #As first step we compute the gradient of the Hamiltonian dF2.
            #Secondly, we compute the product between the contmat (coming from the contact_tensorhd function)
            #And Finally we subtract the value of the Hamiltonian on the Reeb component.
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] #1
            # DOMANDA: io ho messo [0] perche' altrimenti mi dava dei problemi, ha senso?
            horizontal_field = torch.einsum('kij,kj->ki',contmat,dF2)
        if self.field_type != 'vertical':
            vertical_field[:,m-1] = - F2[:,0]
        #if self.field_type != 'solenoidal':
        #    dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
        #    conservative_field = dF1 @ torch.eye(*self.M.shape)

        #if self.field_type != 'conservative':
            #This part computes the Hamiltonian vector field of the related Hamiltonian.
            #As first step we compute the gradient of the Hamiltonian dF2.
            #Secondly, we compute the product between the contmat (coming from the contact_tensorhd function)
            #And Finally we subtract the value of the Hamiltonian on the Reeb component.
        #    dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] #1
            # DOMANDA: io ho messo [0] perche' altrimenti mi dava dei problemi, ha senso?
        #    solenoidal_field = torch.einsum('kij,kj->ki',contmat,dF2)
        #    solenoidal_field[:,m-1] = solenoidal_field[:,m-1] - F2[:,0] #2 é solo il finale nel caso 3d nel caso 2n+1 é 2n
            # DOMANDA: stessa cosa della domanda precedente F2 dovrebbe essere un vettore di valori della funzione
            #          hamiltoniana (ogni valore e' associato ad un punto (q,p,s))

        #if separate_fields:
        #    return [conservative_field, solenoidal_field]
        
        return horizontal_field + vertical_field

###############################
# remaining from the old code #
###############################
    
    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:n-1], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

#######################
# Time dependent case #
#######################
class HNN_timedep(torch.nn.Module):
    # the x vector is decomposed into
    # x=[q1,...,qn,p1,..pn,s,t]
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN_timedep, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.field_type = field_type

    def forward(self, x):
        # traditional forward passliv
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)
    
    def contact_tensorhd(self, x):
        # Canonical coordinates: 2 m + 2
        # (q1,...,qm,p1,...,pm,s)
        m=(x[0,:].size()[0])//2
        # here i can define m as the tensor of dimension and then define from that l.
        p=x[:,m:2*m] #cut the momenta
        l=p.size()[0] #number of entries
        mat0=torch.eye(m)
        mat=torch.zeros(l,2*m+1,2*m+1)
        for i in range(l):
            mat[i,m:2*m,:m]=-mat0
            mat[i,:m,m:2*m]= mat0
            mat[i,2*m,m:2*m] = p[i,:]
            mat[i,m:2*m,2*m] = - p[i,:]      
        return mat
    
    def rk4_time_derivative(self, x, t, dt):
        return rk4(fun=self.time_derivative, y0=x, t=t, dt=dt)

    def time_derivative(self, x, t, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''      
        tt=t.reshape(len(t),1)
        y=torch.cat((tt,x),1) #this has dimension 6:no idea
        m=x.shape[1]
        if self.baseline:
            return self.differentiable_model(y)
        
        ###########################################
        # Splitting between times and phase space #
        # coordinates.                            #
        ###########################################

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(y) # traditional forward pass
        horizontal_field = torch.zeros_like(x) # start out with both components set to 0
        vertical_field = torch.zeros_like(x)
        contmat=self.contact_tensorhd(x)

        if self.field_type != 'horizontal':
            #This part computes the Hamiltonian vector field of the related Hamiltonian.
            #As first step we compute the gradient of the Hamiltonian dF2.
            #Secondly, we compute the product between the contmat (coming from the contact_tensorhd function)
            #And Finally we subtract the value of the Hamiltonian on the Reeb component.
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] #1
            # DOMANDA: io ho messo [0] perche' altrimenti mi dava dei problemi, ha senso?
            horizontal_field = torch.einsum('kij,kj->ki',contmat,dF2)
        if self.field_type != 'vertical':
            vertical_field[:,m-1] = - F2[:,0]

        return horizontal_field + vertical_field