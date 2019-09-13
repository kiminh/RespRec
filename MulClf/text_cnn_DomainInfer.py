import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

#from https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()


class DomainTextCNN(object):

    #main enter
    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, num_domains, num_latent_domains, 
            l2_reg_lambda, dist_type = "Dirichlet", dist_configure="normal_elu", is_use_adv=False):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.input_d = tf.placeholder(tf.int32, [None, num_domains], name = "input_d")
        self.input_d_idx = tf.placeholder(tf.int32, [None, 1], name = "input_d_indx")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.adv_lambda = tf.placeholder(tf.float32, shape=[], name="adv_lambda")

        self.num_filters_total = num_filters * len(filter_sizes)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.is_use_adv = is_use_adv
        l2_loss = tf.constant(0.0)

        with tf.variable_scope("embedding"):
            self.emb_W = tf.get_variable(
                name="lookup_emb",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=True
                )
            embedded_chars = tf.nn.embedding_lookup(self.emb_W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        
        #cnn+pooling
        #shared
        pub_h_pool = self.cnn("shared", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)
        #private
        all_pri_h_pool = []
        for i in range( num_latent_domains ):
            all_pri_h_pool.append(
                self.cnn("pri-%s" % i, self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)
                )
        all_pri_h_pool = tf.stack(all_pri_h_pool, axis = 1) #[batch_size, num_domains, feature]

        hidden_size = 300
        with tf.variable_scope("domain_infer") as scope:
            # p(alpha | x) for test
            x_feature_p = self.cnn("domain_infer_p", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)
            x_h1 = tf.layers.dense(
                inputs = x_feature_p,
                units = hidden_size,
                activation = tf.nn.relu6
                )
            dist_p, samples_p, alpha_p = self.make_distribution(
                h1 = x_h1,
                num_latent_domains = num_latent_domains,
                dist_type = dist_type,
                dist_conf = dist_configure,
                scope_name = 'p'
                )

            # q(z | x,y,d) for training
            x_feature_q = self.cnn("domain_infer_q", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)

            domain_embedding_size = num_latent_domains
            emb_W_d = tf.get_variable(
                name="lookup_emb_domain",
                shape=[num_domains + 1, domain_embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=True
                )
            emb_d = tf.nn.embedding_lookup(emb_W_d, tf.reshape(self.input_d_idx, [-1]) )

            xyd_cat = tf.concat( [x_feature_q, tf.to_float(self.input_y), emb_d], axis = 1 )
            xyd_h1 = tf.layers.dense(
                inputs = xyd_cat,
                units = hidden_size,
                activation = tf.nn.relu6
                )
            dist_q, samples_q, alpha_q = self.make_distribution(
                h1 = xyd_h1,
                num_latent_domains = num_latent_domains,
                dist_type = dist_type,
                dist_conf = dist_configure,
                scope_name = 'q'
                )
            self.alpha_p = alpha_p
            self.samples_p = samples_p

            kl_losses = dist_p.kl_divergence( dist_q )
            self.kl_loss = tf.reduce_mean(kl_losses, name="kl_loss")

            if num_latent_domains == num_domains:
                cor_pred = tf.cast(
                    tf.equal( tf.argmax(alpha_q, 1), tf.argmax(self.input_d, 1) ),
                    "float"
                    )
                self.domain_acc_train = tf.reduce_mean( cor_pred, name="train_accuracy" )
                cor_pred = tf.cast(
                    tf.equal( tf.argmax(alpha_p, 1), tf.argmax(self.input_d, 1) ),
                    "float"
                    )
                self.domain_acc_test = tf.reduce_mean( cor_pred, name="test_accuracy" )
            else:
                self.domain_acc_train = tf.constant( 0 )
                self.domain_acc_test = tf.constant( 0 )

        if num_latent_domains == num_domains:
            self.y_loss_pretrain, self.y_acc_pretrain, self.y_scores_pretrain = self.calc_loss(
                samples = tf.to_float( self.input_d ),
                all_pri_h_pool = all_pri_h_pool,
                pub_h_pool = pub_h_pool,
                reuse = None,
                )
        else: #shouldnot be used
            self.y_loss_pretrain, self.y_acc_pretrain, self.y_scores_pretrain = self.calc_loss(
                samples = samples_q,
                all_pri_h_pool = all_pri_h_pool,
                pub_h_pool = pub_h_pool,
                reuse = None,
                )

        self.y_loss_train, self.y_acc_train, self.y_scores_train = self.calc_loss(
            samples = samples_q,
            all_pri_h_pool = all_pri_h_pool,
            pub_h_pool = pub_h_pool,
            reuse = True,
            )

        self.y_loss_test, self.y_acc_test, self.y_scores_test = self.calc_loss(
            samples = samples_p,
            all_pri_h_pool = all_pri_h_pool,
            pub_h_pool = pub_h_pool,
            reuse = True,
            )


    def calc_loss(self, samples, all_pri_h_pool, pub_h_pool, reuse = None):

        with tf.variable_scope("labels_loss", reuse = reuse) as scope:
            pri_h = tf.matmul(
                tf.cast( tf.expand_dims(samples, axis=1), "float32"),
                all_pri_h_pool
            )
            pri_h = tf.reshape(pri_h, shape = [-1, self.num_filters_total])
            if self.is_use_adv:
                h = tf.concat( [pri_h, pub_h_pool], axis = 1 )
            else:
                h = pri_h

            self.h = h
            # input_dim = num_filters_total
            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h, self.dropout_keep_prob)
                pub_h_drop = tf.nn.dropout(pub_h_pool, self.dropout_keep_prob)

            if self.is_use_adv:
                with tf.variable_scope("adv_loss", reuse = reuse) as scope:
                    pub_feature = flip_gradient(pub_h_drop, self.adv_lambda)
                    a_h1 = tf.layers.dense(
                        inputs = pub_feature,
                        units = 50,
                        activation = tf.nn.relu6,
                        name = "adv_h1",
                        reuse = reuse,
                        )
                    d_scores = tf.layers.dense(
                        inputs = a_h1,
                        units = self.num_domains,
                        activation = None,
                        name = "output",
                        reuse = reuse,
                        )
                    with tf.name_scope("loss"):
                        losses = tf.nn.softmax_cross_entropy_with_logits(
                            logits = d_scores,
                            labels = self.input_d,
                            )
                        d_loss = tf.reduce_mean(losses, name="task_loss")
                    with tf.name_scope("accuracy"):
                        d_pred = tf.argmax(d_scores, 1, name="predictions")
                        cor_pred = tf.cast(
                            tf.equal( d_pred, tf.argmax(self.input_d, 1) ),
                            "float"
                            )
                        d_accuracy = tf.reduce_mean( cor_pred, name="accuracy" )

            hidden_size = 300
            with tf.variable_scope("label", reuse = reuse) as scope:
                h1 = tf.layers.dense(
                    inputs = h_drop,
                    units = hidden_size,
                    activation = tf.nn.relu6,
                    name = "h1",
                    reuse = reuse,
                    )
                y_scores = tf.layers.dense(
                    inputs = h1,
                    units = self.num_classes,
                    activation = None,
                    name = "output",
                    reuse = reuse,
                    )
                y_scores_softmax = tf.nn.softmax(y_scores)

                # CalculateMean cross-entropy loss
                with tf.name_scope("loss"):
                    losses = tf.nn.softmax_cross_entropy_with_logits(
                        logits = y_scores,
                        labels = self.input_y,
                        )
                    y_loss = tf.reduce_mean(losses, name="task_loss")

                with tf.name_scope("accuracy"):
                    y_pred = tf.argmax(y_scores, 1, name="predictions")
                    cor_pred = tf.cast(
                        tf.equal( y_pred, tf.argmax(self.input_y, 1) ),
                        "float"
                        )
                    y_accuracy = tf.reduce_mean( cor_pred, name="accuracy" )
            
            if self.is_use_adv:
                return y_loss + d_loss, y_accuracy, y_scores_softmax
            else:
                return y_loss, y_accuracy, y_scores_softmax
    

    def make_distribution(self, h1, num_latent_domains, dist_type, dist_conf, scope_name):
        with tf.variable_scope(scope_name) as scope:
            if dist_type == "Dirichlet":

                if dist_conf == 'normal_elu':
                    alpha = tf.layers.dense(
                        inputs = h1,
                        units = num_latent_domains,
                        activation = tf.nn.elu
                        ) # shape=[batch_size , num_domains]
                    alpha = alpha + 1 + 1e-6
                elif dist_conf == 'sparsity_sigmoid':
                    alpha = tf.layers.dense(
                        inputs = h1,
                        units = num_latent_domains,
                        activation = tf.nn.sigmoid
                        )
                    alpha_0 = tf.layers.dense(
                        inputs = h1,
                        units = 1,
                        activation = tf.exp
                        )
                    alpha = alpha * alpha_0
                elif dist_conf == "sparsity_softmax":
                    alpha = tf.layers.dense(
                        inputs = h1,
                        units = num_latent_domains,
                        activation = tf.nn.softmax
                        )
                    alpha_0 = tf.layers.dense(
                        inputs = h1,
                        units = 1,
                        activation = tf.exp
                        )
                    alpha = alpha * alpha_0
                else:
                    assert(False)

                dist = tf.distributions.Dirichlet( alpha )
                samples = dist.sample( ) # same size as alpha_o[]
                return dist, samples, alpha

            elif dist_type == "Gamma":

                alpha = tf.layers.dense(
                    inputs = h1,
                    units = num_latent_domains,
                    activation = tf.nn.elu
                    ) # shape=[batch_size , num_domains]
                alpha = alpha + 1 + 1e-6

                beta = tf.layers.dense(
                    inputs = h1,
                    units = num_latent_domains,
                    activation = tf.nn.elu
                    ) # shape=[batch_size , num_domains]
                beta = beta + 1 + 1e-6

                dist = tf.distributions.Gamma( alpha, beta )
                samples = dist.sample( )
                samples = tf.nn.softmax( samples )
                return dist, samples, alpha

            elif dist_type == "Beta":

                alpha = tf.layers.dense(
                    inputs = h1,
                    units = num_latent_domains,
                    activation = tf.nn.elu
                    ) # shape=[batch_size , num_domains]
                alpha = alpha + 1 + 1e-6

                beta = tf.layers.dense(
                    inputs = h1,
                    units = num_latent_domains,
                    activation = tf.nn.elu
                    ) # shape=[batch_size , num_domains]
                beta = beta + 1 + 1e-6

                dist = tf.distributions.Beta( alpha, beta )
                samples = dist.sample( )
                samples = tf.nn.softmax( samples )
                return dist, samples, alpha

            else:
                print("Distribution Type not Implement!")
                assert(False)


    def cnn(self, scope_number, embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters):
        with tf.variable_scope("cnn%s" % scope_number):
        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(
                        name="W",
                        shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                        )
                    b = tf.get_variable(
                        name="b",
                        shape=[num_filters],
                        initializer=tf.constant_initializer(0.1)
                        )
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            pooled = tf.concat(pooled_outputs, 3)
            return tf.reshape(pooled, [-1, num_filters * len(filter_sizes)])
