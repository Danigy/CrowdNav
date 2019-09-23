'''
    @name:      state_collector.py
    @brief:     This class provides custom policies used in stable-baselines library
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

from stable_baselines.common.policies import *

NS="sim1"

# Added by Ronja Gueldenring
def conv1d(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NWC', one_dim_bias=False):
    """
    Creates a 1d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size: (int) The filter size
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if data_format == 'NWC':
        channel_ax = 2
        bshape = [1, 1, n_filters]
    elif data_format == 'NCW':
        channel_ax = 0
        bshape = [1, n_filters, 1]
    else:
        raise NotImplementedError

    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_size, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv1d(input_tensor, weight, stride=stride, padding=pad, data_format=data_format)

def laser_cnn(state, **kwargs):
    """
    1D Conv Network

    :param state: (TensorFlow Tensor) Scan input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    scan = tf.squeeze(state, axis=1)
    activ = tf.nn.relu
    layer_1 = activ(
        conv1d(scan, 'c1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs))
    # layer_2 = activ(conv1d(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv1d(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    layer_3 = activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    return activ(linear(layer_3, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))

class CNN1DPolicy(FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Polar Representation
    """
    def __init__(self, *args, **kwargs):
        super(CNN1DPolicy, self).__init__(*args, n_env=1, n_steps=1, n_batch=None, cnn_extractor=laser_cnn, feature_extraction="cnn", **kwargs)

def laser_cnn_multi_input(state, **kwargs):
    """
    1D Conv Network

    :param state: (TensorFlow Tensor) state input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    scan = tf.squeeze(state[:, : , 0:kwargs['laser_scan_len'] , :], axis=1)
    goal = tf.squeeze(state[:, :, kwargs['laser_scan_len']:, :], axis=1)
    goal = tf.squeeze(goal, axis=2)
    # goal = tf.math.multiply(goal, 6)

    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv1d(scan, 'c1d_1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = activ(conv1d(layer_1, 'c1d_2', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = conv_to_fc(layer_2)
    layer_3 = activ(linear(layer_2, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    temp = tf.concat([layer_3, goal], 1)
    # print_goal = tf.print(tf.shape(layer_1))
    # with tf.control_dependencies([print_goal]):
    layer_4 = activ(linear(temp, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return layer_4


class CNN1DPolicy_multi_input(FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Raw Data Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["laser_scan_len"] =  360
        super(CNN1DPolicy_multi_input, self).__init__(*args, **kwargs, cnn_extractor=laser_cnn_multi_input, feature_extraction="cnn")

def nature_cnn_multi_input(state, **kwargs):
    """
    CNN from Nature paper. state[0:-1, :, :] is the input image, while state[-1, :, :]
    provides additional information, that is concenated later with the output pf layer 3.
    It can be additional non-image information.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    img = state[:, 0:-1 , : , :]
    vel = state[:, -1, 0:kwargs['multi_input_size'], 0]
    vel = tf.math.multiply(vel, kwargs['max_image_value'])
    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv(img, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs_conv))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = conv_to_fc(layer_3)
    concat_img_goal = tf.concat([layer_3, vel], 1)
    # For printing uncomment: velocity
    # print_vel = tf.print(vel)
    # with tf.control_dependencies([print_vel]):
    layer_4 = activ(linear(concat_img_goal, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return layer_4

class CnnPolicy_multi_input_vel(FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel, self).__init__(*args, **kwargs, cnn_extractor=nature_cnn_multi_input, feature_extraction="cnn")

def cnn_multi_input(state, **kwargs):
    """
    CNN from Nature paper. state[0:-1, :, :] is the input image, while state[-1, :, :]
    provides additional information, that is concenated later with the output pf layer 3.
    It can be additional non-image information.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    img = state[:, 0:-1 , : , :]
    vel = state[:, -1, 0:kwargs['multi_input_size'], -1]
    vel = tf.math.multiply(vel, kwargs['max_image_value'])
    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv(img, 'c1', n_filters=64, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs_conv))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_4 = conv_to_fc(layer_4)
    concat_img_goal = tf.concat([layer_4, vel], 1)
    # For printing uncomment: velocity
    # print_vel = tf.print(vel)
    # with tf.control_dependencies([print_vel]):
    layer_4 = activ(linear(concat_img_goal, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    layer_5 = activ(linear(layer_4, 'fc2', n_hidden=216, init_scale=np.sqrt(2)))
    return layer_5

class CnnPolicy_multi_input_vel2(FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel2, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")

# Same as CnnPolicy_multi_input_vel2. Is necessary for executing agents, that were trained on this policy-name
class CnnPolicy_multi_input_vel3(FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel3, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")
