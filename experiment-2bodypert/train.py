import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN_timedep
from data import get_dataset, hamiltonian_fn
from utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=6, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=500, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='dimension of batch')
    parser.add_argument('--nonlinearity', default='snake', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=4000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wedight decay for Adams') 
    parser.add_argument('--name', default='2bodypert', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='contact', type=str, help='type of vector field to learn')
    parser.add_argument('--hamiltoniantrain', dest='hamint', action='store_true', help='Splitting Hamiltonian Training')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim-1 if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN_timedep(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=args.weight_decay)

  # arrange data
  data = get_dataset(hamint=args.hamint, save_dir=args.save_dir, seed=args.seed)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])
  print('Initialized Dataset')
  print('Number of training data: ', data['x'].shape[0])
  print('Number of test data:     ', data['test_x'].shape[0])
    
  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
    # train step
    pind = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.rk4_time_derivative(x=x[pind,1:], t=x[pind,0]) if args.use_rk4 else model.time_derivative(x=x[pind,1:], t=x[pind,0])
    loss = L2_loss(dxdt[pind], dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # run test data
    test_pind = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_dxdt_hat = model.rk4_time_derivative(test_x[test_pind,1:],test_x[test_pind,0]) if args.use_rk4 else model.time_derivative(x=test_x[test_pind,1:],t=test_x[test_pind,0])
    test_loss = L2_loss(test_dxdt[test_pind], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
  
  del pind, test_pind, test_dxdt_hat, dxdt_hat, test_loss, loss

  N_max=5000
  if N_max>len(test_dxdt):
    N_max=len(test_dxdt)
  #
  if args.verbose:
    print('Final train loss computations: ')
  dxdt_hat = model.time_derivative(x=x[:,1:], t=x[:,0])
  train_dist = (dxdt - dxdt_hat)**2
  del dxdt_hat
  if args.verbose:
    print('- train loss done')
  test_dxdt_hat = model.time_derivative(x=test_x[:N_max,1:],t=test_x[:N_max,0])
  test_dist = (test_dxdt[:N_max] - test_dxdt_hat)**2
  if args.verbose:
    print("- test loss done")
  del test_dxdt_hat
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-hamint' + label if args.hamint else label
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
