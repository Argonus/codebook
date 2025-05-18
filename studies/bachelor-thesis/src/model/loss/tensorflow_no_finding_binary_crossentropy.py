import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='custom_losses')
class NoFindingBinaryCrossentropy(tf.keras.losses.Loss):
    """
    Custom loss function that enforces the logical relationship between "No Finding" class and other conditions.
    When "No Finding" is true, other conditions should not be predicted, and vice versa.
    """
    def __init__(self, no_finding_idx, with_sigmoid=False, sigmoid_scale=10.0, lambda_value=0.2, from_logits=False, label_smoothing=0.00, **kwargs):
        super(NoFindingBinaryCrossentropy, self).__init__(**kwargs)
        self.no_finding_idx = no_finding_idx
        self.with_sigmoid = with_sigmoid
        self.sigmoid_scale = sigmoid_scale
        self.lambda_value = lambda_value
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    
    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        
        other_conditions_pred = tf.reduce_sum(y_pred, axis=1) - y_pred[:, self.no_finding_idx]        
        true_no_finding = tf.cast(y_true[:, self.no_finding_idx], tf.float32)
        
        loss_1_pred_val = self._maybe_apply_sigmoid(other_conditions_pred)
        inconsistency_loss_1 = true_no_finding * loss_1_pred_val

        loss_2_pred_val = self._maybe_apply_sigmoid(1 - (1 - tf.clip_by_value(other_conditions_pred, 0, 1)))
        inconsistency_loss_2 = (1 - true_no_finding) * loss_2_pred_val

        inconsistency_loss = tf.reduce_mean(inconsistency_loss_1 + inconsistency_loss_2)
        return bce_loss + self.lambda_value * inconsistency_loss

    def _maybe_apply_sigmoid(self, pred):
        if self.with_sigmoid:
            return tf.sigmoid(self.sigmoid_scale * pred)

        return pred
    
    def get_config(self):
        config = super(NoFindingBinaryCrossentropy, self).get_config()
        config.update({
            'no_finding_idx': self.no_finding_idx,
            'lambda_value': self.lambda_value,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config
