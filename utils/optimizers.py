import numpy as np

def adam(x, dx, config=None):
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  config['t'] += 1
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx**2
  m_hat = config['m'] / (1 - config['beta1']**config['t'])
  v_hat = config['v'] / (1 - config['beta2']**config['t'])
  next_x = x - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

  return next_x, config