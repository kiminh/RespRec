  var_loss = 0.5 * tf.reduce_sum(tf.square(weights - tf.reduce_mean(weights)))
  # input(tf.gradients(var_loss, wt_var_list))
  ubs_loss += var_loss



Synthesizing Robust Adversarial Examples

Adversarial Time-to-Event Modeling
