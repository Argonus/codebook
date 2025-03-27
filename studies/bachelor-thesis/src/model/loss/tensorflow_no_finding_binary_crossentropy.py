import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='custom_losses')
class NoFindingBinaryCrossentropy(tf.keras.losses.Loss):
    """
    Custom loss function that enforces the logical relationship between "No Finding" class and other conditions.
    When "No Finding" is true, other conditions should not be predicted, and vice versa.
    """
    def __init__(self, no_finding_idx, lambda_value=0.2, from_logits=False, label_smoothing=0.01, **kwargs):
        super(NoFindingBinaryCrossentropy, self).__init__(**kwargs)
        self.no_finding_idx = no_finding_idx
        self.lambda_value = lambda_value
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    
    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        
        other_conditions_pred = tf.reduce_sum(y_pred, axis=1) - y_pred[:, self.no_finding_idx]        
        true_no_finding = tf.cast(y_true[:, self.no_finding_idx], tf.float32)
        
        inconsistency_loss_1 = true_no_finding * other_conditions_pred
        inconsistency_loss_2 = (1 - true_no_finding) * (1 - tf.clip_by_value(other_conditions_pred, 0, 1))
        inconsistency_loss = tf.reduce_mean(inconsistency_loss_1 + inconsistency_loss_2)
        
        return bce_loss + self.lambda_value * inconsistency_loss
    
    def get_config(self):
        config = super(NoFindingBinaryCrossentropy, self).get_config()
        config.update({
            'no_finding_idx': self.no_finding_idx,
            'lambda_value': self.lambda_value,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config
