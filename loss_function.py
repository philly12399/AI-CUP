import tensorflow as tf
import tensorflow.keras.backend as kb
def F_measure_loss(y_actual,y_pred):
    y_actual = tf.dtypes.cast(y_actual, tf.float32)
    A = kb.sum(y_actual)
    B = kb.sum(y_pred)
    C = kb.dot(kb.reshape(y_actual, (1,-1)), kb.reshape(y_pred, (-1,1)))
    C = kb.reshape(C, ())
    precision = C/B
    recall = C/A
    f_measure = 2*precision*recall/(precision + recall)
    return 1 - f_measure
"""
y_pred = tf.constant([0.3, 0.1, 0.8, 0.2])
y_actual= tf.constant([0.0, 0.0, 1.0, 0.0])
print(F_measure_loss(y_actual, y_pred))
"""