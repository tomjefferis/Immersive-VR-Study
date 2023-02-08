import tensorflow as tf
from tensorflow.keras import layers

class CBAM(layers.Layer):
    def __init__(self, channel_attention_module=True, spatial_attention_module=True,
                 reduction_ratio=1, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channel_attention_module = channel_attention_module
        self.spatial_attention_module = spatial_attention_module
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channel_axis = -1
        filters = input_shape[channel_axis]
        
        self.channel_average_pooling = layers.GlobalAveragePooling2D(data_format='channels_last')
        self.channel_max_pooling = layers.GlobalMaxPooling2D(data_format='channels_last')
        self.fc1 = layers.Dense(filters // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(filters, activation='sigmoid')
        
        if self.spatial_attention_module:
            self.conv1 = layers.Conv2D(filters // self.reduction_ratio, kernel_size=4, strides=1,
                                       activation='relu', padding='same')
            self.conv2 = layers.Conv2D(filters // self.reduction_ratio, kernel_size=4, strides=1,
                                       activation='sigmoid', padding='same')
        
        super(CBAM, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        attention_feature = inputs
        if self.channel_attention_module:
            channel_attention = self.channel_average_pooling(attention_feature)
            channel_attention = self.fc1(channel_attention)
            channel_attention = self.fc2(channel_attention)
            channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)
            attention_feature = attention_feature * channel_attention
        
        if self.spatial_attention_module:
            spatial_attention = self.conv1(attention_feature)
            spatial_attention = self.conv2(spatial_attention)
            attention_feature = attention_feature * spatial_attention
        
        return attention_feature
