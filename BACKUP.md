  var_loss = 0.5 * tf.reduce_sum(tf.square(weights - tf.reduce_mean(weights)))
  # input(tf.gradients(var_loss, wt_var_list))
  ubs_loss += var_loss


Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting

Synthesizing Robust Adversarial Examples

Adversarial Time-to-Event Modeling
