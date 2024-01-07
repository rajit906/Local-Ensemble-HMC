'''
Code to extend TensorFlow Porbabilit's Source Code for HMC to LEPHMC (Still buggy!)
'''

from tensorflow_probability.python.mcmc import kernel as kernel_base
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import collections
import abc
import six
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.experimental.mcmc import preconditioning_utils as pu
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import UncalibratedPreconditionedHamiltonianMonteCarlo, UncalibratedPreconditionedHamiltonianMonteCarloKernelResults, _prepare_args, _compute_log_acceptance_correction, PreconditionedHamiltonianMonteCarlo

__all__ = [
    'LeapfrogIntegrator',
    'SimpleLeapfrogIntegrator',
    'process_args',
]


@six.add_metaclass(abc.ABCMeta)
class LeapfrogIntegrator(object):
  @abc.abstractmethod
  def __call__(self, momentum_parts, state_parts, cov, target=None,
               target_grad_parts=None, kinetic_energy_fn=None, name=None):
    raise NotImplementedError('Integrate logic not implemented.')


class LocalLeapfrogIntegrator(LeapfrogIntegrator): #TODO: Change Name

  def __init__(self, target_fn, step_sizes, num_steps):
    self._target_fn = target_fn
    self._step_sizes = step_sizes
    self._num_steps = num_steps

  @property
  def target_fn(self):
    return self._target_fn

  @property
  def step_sizes(self):
    return self._step_sizes

  @property
  def num_steps(self):
    return self._num_steps

  def __call__(self,
               momentum_parts,
               state_parts,
               cov,
               target=None,
               target_grad_parts=None,
               kinetic_energy_fn=None,
               name=None):
    with tf.name_scope(name or 'leapfrog_integrate'):
      [
          momentum_parts,
          state_parts,
          target,
          target_grad_parts,
      ] = process_args(
          self.target_fn,
          momentum_parts,
          state_parts,
          target,
          target_grad_parts)

      if kinetic_energy_fn is None:
        get_velocity_parts = lambda x: x
      else:
        def get_velocity_parts(half_next_momentum_parts):
          _, velocity_parts = mcmc_util.maybe_call_fn_and_grads(
              kinetic_energy_fn, half_next_momentum_parts)
          return velocity_parts


      half_next_momentum_parts = [
          v + _multiply(0.5 * eps, g, dtype=v.dtype) #Should this be + or -?
          for v, eps, g
          in zip(momentum_parts, self.step_sizes, target_grad_parts)]

      [
          _,
          next_half_next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      ] = tf.while_loop(
          cond=lambda i, *_: i < self.num_steps,
          body=lambda i, *args: [i + 1] + list(_one_step(
              self.target_fn, self.step_sizes, get_velocity_parts, cov, *args)),
          loop_vars=[
              tf.zeros_like(self.num_steps, name='iter'),
              half_next_momentum_parts,
              state_parts,
              target,
              target_grad_parts,
          ])

      next_momentum_parts = [
          v - _multiply(0.5 * eps, g, dtype=v.dtype)
          for v, eps, g
          in zip(next_half_next_momentum_parts,
                 self.step_sizes,
                 next_target_grad_parts)
      ]

      return (
          next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      )


def _one_step(
    target_fn,
    step_sizes,
    get_velocity_parts,
    cov,
    half_next_momentum_parts,
    state_parts,
    target,
    target_grad_parts):

  with tf.name_scope('leapfrog_integrate_one_step'):

    velocity_parts = get_velocity_parts(half_next_momentum_parts)
    next_state_parts = []
    for state_part, eps, velocity_part in zip(
        state_parts, step_sizes, velocity_parts):
      next_state_parts.append(
          state_part + _multiply(eps, tf.linalg.matvec(cov, velocity_part), dtype=state_part.dtype))
    [next_target, next_target_grad_parts] = mcmc_util.maybe_call_fn_and_grads(
        target_fn, next_state_parts)
    if any(g is None for g in next_target_grad_parts):
      raise ValueError(
          'Encountered `None` gradient.\n'
          '  state_parts: {}\n'
          '  next_state_parts: {}\n'
          '  next_target_grad_parts: {}'.format(
              state_parts,
              next_state_parts,
              next_target_grad_parts))
    tensorshape_util.set_shape(next_target, target.shape)
    for ng, g in zip(next_target_grad_parts, target_grad_parts):
      tensorshape_util.set_shape(ng, g.shape)

    next_half_next_momentum_parts = [
        v + _multiply(eps, g, dtype=v.dtype)  # pylint: disable=g-complex-comprehension
        for v, eps, g
        in zip(half_next_momentum_parts, step_sizes, next_target_grad_parts)]

    return [
        next_half_next_momentum_parts,
        next_state_parts,
        next_target,
        next_target_grad_parts,
    ]


def process_args(target_fn, momentum_parts, state_parts,
                 target=None, target_grad_parts=None):
  """Sanitize inputs to `__call__`."""
  with tf.name_scope('process_args'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None or target_grad_parts is None:
      [target, target_grad_parts] = mcmc_util.maybe_call_fn_and_grads(
          target_fn, state_parts)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
      target_grad_parts = [
          tf.convert_to_tensor(
              g, dtype_hint=tf.float32, name='target_grad_part')
          for g in target_grad_parts]
    return momentum_parts, state_parts, target, target_grad_parts


def _multiply(tensor, state_sized_tensor, dtype):
  result = tf.cast(tensor, dtype) * tf.cast(state_sized_tensor, dtype)
  tensorshape_util.set_shape(result, state_sized_tensor.shape)
  return result

class UncalibratedLocalEnsemblePreconditionedHMC(hmc.UncalibratedHamiltonianMonteCarlo):
  @mcmc_util.set_doc(hmc.UncalibratedHamiltonianMonteCarlo.__init__)
  def __init__(self,
               target_log_prob_fn,
               num_leapfrog_steps,
               step_size,
               momentum_distribution=None,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):
    super(UncalibratedLocalEnsemblePreconditionedHMC, self).__init__(
        target_log_prob_fn,
        step_size,
        num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        store_parameters_in_results=store_parameters_in_results,
        experimental_shard_axis_names=experimental_shard_axis_names,
        name=name)
    self._parameters['momentum_distribution'] = momentum_distribution
    self._parameters.pop('seed', None)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']

  @property
  def num_leapfrog_steps(self):
    return self._parameters['num_leapfrog_steps']

  @property
  def momentum_distribution(self):
    return self._parameters['momentum_distribution']

  def compute_local_covariance(self, pos, current_target_log_prob_grad_parts, stepsize, N_walkers, N_steps):
    pos_initial = pos
    local_samples = pos_initial
    for i in range(N_walkers):
      pos = pos_initial
      for j in range(N_steps): #TODO: Parallelize this.
        pos -= stepsize * current_target_log_prob_grad_parts + tf.sqrt(2 * stepsize) * tfd.MultivariateNormalDiag(loc=[0., 0.]).sample()
        local_samples = tf.concat([local_samples,pos],axis=0)
    local_samples = tf.reshape(local_samples, shape=(-1,2))
    local_samples -= tf.reduce_mean(local_samples, axis=0)
    cov = 1/(N_walkers*N_steps) * tf.matmul(tf.transpose(local_samples), local_samples)
    return cov

  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'lphmc', 'one_step')):
      if self._store_parameters_in_results:
        step_size = previous_kernel_results.step_size
        num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
        momentum_distribution = previous_kernel_results.momentum_distribution
      else:
        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps
        momentum_distribution = self.momentum_distribution

      [
          current_state_parts,
          step_sizes,
          momentum_distribution,
          current_target_log_prob,
          current_target_log_prob_grad_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          step_size,
          momentum_distribution,
          previous_kernel_results.target_log_prob,
          previous_kernel_results.grads_target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped,
          experimental_shard_axis_names=self.experimental_shard_axis_names)

      cov = self.compute_local_covariance(current_state_parts, current_target_log_prob_grad_parts, tf.constant([self.step_size]), 5, 10)
      momentum_distribution = tfd.JointDistributionSequential([tfd.MultivariateNormalTriL(loc = [0., 0.], scale_tril = tf.linalg.cholesky(cov))]) # Should it be cov or cov cov.T
      seed = samplers.sanitize_seed(seed)
      current_momentum_parts = list(momentum_distribution.sample(seed=seed))
      momentum_log_prob = getattr(momentum_distribution,
                                  '_log_prob_unnormalized',
                                  momentum_distribution.log_prob)
      kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)

      integrator = LocalLeapfrogIntegrator(
          self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,
      ] = integrator(
          current_momentum_parts,
          current_state_parts,
          cov,
          target=current_target_log_prob,
          target_grad_parts=current_target_log_prob_grad_parts,
          kinetic_energy_fn=kinetic_energy_fn)
      if self.state_gradients_are_stopped:
        next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]
      new_kernel_results = previous_kernel_results._replace(
          log_acceptance_correction=_compute_log_acceptance_correction(
              kinetic_energy_fn, current_momentum_parts,
              next_momentum_parts),
          target_log_prob=next_target_log_prob,
          grads_target_log_prob=next_target_log_prob_grad_parts,
          initial_momentum=current_momentum_parts,
          final_momentum=next_momentum_parts,
          seed=seed,
      )

      return maybe_flatten(next_state_parts), new_kernel_results

  @mcmc_util.set_doc(hmc.HamiltonianMonteCarlo.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'lphmc', 'bootstrap_results')):
      result = super(UncalibratedLocalEnsemblePreconditionedHMC,
                     self).bootstrap_results(init_state)

      state_parts, _ = mcmc_util.prepare_state_parts(init_state,
                                                     name='current_state')
      target_log_prob = result.target_log_prob
      if (not self._store_parameters_in_results or
          self.momentum_distribution is None):
        momentum_distribution = pu.make_momentum_distribution(
            state_parts, ps.shape(target_log_prob),
            shard_axis_names=self.experimental_shard_axis_names
        )
      else:
        momentum_distribution = pu.maybe_make_list_and_batch_broadcast(
            self.momentum_distribution, ps.shape(target_log_prob))
      result = UncalibratedPreconditionedHamiltonianMonteCarloKernelResults(
          **result._asdict(),
          momentum_distribution=momentum_distribution)
    return result


class LocalEnsemblePreconditionedHMC(hmc.HamiltonianMonteCarlo):
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               momentum_distribution=None,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):
    if step_size_update_fn and store_parameters_in_results:
      raise ValueError('It is invalid to simultaneously specify '
                       '`step_size_update_fn` and set '
                       '`store_parameters_in_results` to `True`.')
    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedLocalEnsemblePreconditionedHMC(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            momentum_distribution=momentum_distribution,
            name=name or 'lphmc_kernel',
            experimental_shard_axis_names=experimental_shard_axis_names,
            store_parameters_in_results=store_parameters_in_results))
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters.pop('seed', None)
    self._parameters['step_size_update_fn'] = step_size_update_fn

