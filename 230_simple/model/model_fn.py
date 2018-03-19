"""Define the model."""

import tensorflow as tf
import numpy as np

def build_model(mode, inputs, params, is_training):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = inputs['sentence']
    max_length = 50
    #l2_scale = 0.01
    
    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)
        
        # Self Attentive Classification Network implementation
        sentence = tf.layers.dense(sentence, 10, activation = tf.nn.relu)
        # Apply a bidirectional LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        outputs, last_output_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, sentence, 
                dtype=tf.float32, sequence_length=inputs['sentence_lengths']
                ,scope = 'Encoder')
        output = tf.concat(outputs, 2)
        
        W_1 = tf.get_variable('W_1',shape = [2*params.lstm_num_units, params.attention_units],
                             initializer = tf.glorot_uniform_initializer())
        v   = tf.get_variable('v'  ,shape = [params.attention_units,  2*params.lstm_num_units],
                              initializer = tf.glorot_uniform_initializer())
        s = []
        for w in range(max_length):
            u = tf.tanh( tf.matmul( output[:,w,:], (W_1) ) )
            s.append(tf.matmul(u, v))
        
        scores = tf.stack(s, axis=1)
        attention_scores = tf.exp(scores)/tf.reduce_sum(tf.exp(scores), axis =1, keepdims =True)
        # (m, Tx, emb)
        
        context = tf.multiply(output, attention_scores)
        #context_diff = tf.subtract(attention_scores, output)
        attention_layer_output = tf.concat([output, context], axis = 2)
        """
        integration_cell = tf.nn.rnn_cell.BasicLSTMCell(attention_output.get_shape().as_list()[2])
        attention_layer_outputs, last_output_states = tf.nn.bidirectional_dynamic_rnn(
                integration_cell, integration_cell, attention_output, 
                dtype=tf.float32, sequence_length=inputs['sentence_lengths']
                ,scope='Integration_Layer')
        
        attention_layer_output = tf.concat(attention_layer_outputs, 2)
        """
        #meanOutput = tf.reduce_mean(attention_output, axis = 1)
        pool_output5M = tf.nn.pool(input=attention_layer_output, window_shape=[5], pooling_type="MAX", padding="VALID")
        #pool_output5A = tf.nn.pool(input=attention_layer_output, window_shape=[5], pooling_type="AVG", padding="SAME")
        
        mean_output = tf.reduce_mean(pool_output5M, axis = 1)
        
        keep_rate = 1.0
        if is_training:
            keep_rate = 1.0 - params.dropout_rate
        activation_fn = tf.nn.relu
        
        # Compute logits from the output of the LSTM
        layer1_output = tf.layers.dense(mean_output, 5, activation = activation_fn)
        layer1_output = tf.nn.dropout(layer1_output, keep_rate)
        
        logits = tf.layers.dense(layer1_output, 1, activation = activation_fn)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    l2_reg = 0.01
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params, is_training)
        predictions = logits
        #predictions = tf.cast(tf.argmax(logits, -1), tf.int32)
        #labels = tf.reshape(labels, [-1,1])
        #predictions = [1 for i in logits > 0 else 0]
        

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.losses.mean_squared_error(labels =tf.reshape(labels, (-1,1)), predictions =predictions)
    #losses = tf.multiply(tf.to_float(tf.add(9*labels,1)), losses)
    loss = tf.reduce_mean(losses)
    
    #loss += l2_reg*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    #loss += tf.losses.get_regularization_loss()
    
    #accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)#tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            #'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
            #'recall': tf.metrics.recall(labels=labels, predictions=predictions),
            #'precision': tf.metrics.precision(labels=labels, predictions=predictions)
        }
    with tf.variable_scope("preds"):
        pred = { 'predictions': predictions,
                 'labels': labels,
                 'reviews': inputs['sentence']
                 }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)
    
    preds = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="preds")
    
    # Summaries for training
    tf.summary.scalar('loss', loss)
    #tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['loss'] = loss
    #model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec["predictions"] = pred
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    #model_spec['matrix'] = tf.confusion_matrix(labels=labels, predictions=predictions)

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
