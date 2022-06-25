import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        # Create summary writer in log_dir
        with tf.compat.v1.Graph().as_default():
            self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        # Log scalar variable
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value.item())])
        self.writer.add_summary(summary, step)
