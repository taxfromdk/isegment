import tensorflow as tf
#import tensorflow.contrib.slim as slim

tf.compat.v1.disable_eager_execution()

class Network:
    def addConvLayer(s, l, features, kernel_width, dr, last):
        af = None
        if not last:
            af = None
        l = tf.keras.layers.Conv2D(filters=features, kernel_size=(kernel_width, kernel_width), strides=(1,1), padding='same', activation=af, dilation_rate=dr)(l)
        
        """
        def convolution2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
        """
        #if last:
        #   l = slim.conv2d(l, features, [kernel_width, kernel_width], 1, padding='SAME', rate=dilation_rate, activation_fn=None)
        #else:
        #   l = slim.conv2d(l, features, [kernel_width, kernel_width], 1, padding='SAME', rate=dilation_rate)    
        s.receptive_field_range += ((kernel_width-1))/2*dr
        print( l )
        return l

    def __init__(s, IMAGE_HEIGHT, IMAGE_WIDTH, path):
        print("Tensorflow version: ", tf.__version__)
        tf.compat.v1.set_random_seed(42)
        tf.compat.v1.reset_default_graph()

        s.IMAGE_HEIGHT = IMAGE_HEIGHT
        s.IMAGE_WIDTH = IMAGE_WIDTH
        s.path = path

        with tf.compat.v1.name_scope("input"):
            s.x_uint8 = tf.compat.v1.placeholder(tf.uint8, shape=(None, s.IMAGE_HEIGHT, s.IMAGE_WIDTH, 3), name="x")
            print(s.x_uint8)    
            #range 0 to 1
            x_float32 = tf.cast(s.x_uint8, tf.float32)/255.0
            #take a grayscale copy
            x_gray = tf.image.rgb_to_grayscale(x_float32)
            #move to raange -1 to 1
            x_float32 = (x_float32-0.5)*2.0
            print("x_float32",x_float32)
            print("x_gray",x_gray)

            s.lr = tf.compat.v1.placeholder(tf.float32, shape=(), name="learning_rate")
            s.reg = tf.compat.v1.placeholder(tf.float32, shape=(), name="regularization")
            s.amp = tf.compat.v1.placeholder(tf.float32, shape=(), name="regularization")


        s.receptive_field_range = 0 
            
        with tf.compat.v1.name_scope("convolutions"):
            
            l = x_float32
            
            l = s.addConvLayer(l, 16*1, 3, 2, False)
            l = s.addConvLayer(l, 16*2, 3, 2, False)
            l = s.addConvLayer(l, 16*3, 3, 2, False)
            l = s.addConvLayer(l, 16*4, 3, 2, False)
            l = s.addConvLayer(l, 16*5, 3, 2, False)
            l = s.addConvLayer(l, 16*6, 3, 2, False)
            l = s.addConvLayer(l,  1, 3, 2, True)

            response = tf.reshape(l, (-1, s.IMAGE_HEIGHT, s.IMAGE_WIDTH,1))

        s.receptive_field_range = int(s.receptive_field_range)
        print("receptive_field_range", s.receptive_field_range)

        red   = tf.clip_by_value(response*s.amp, 0.0, 1.0)
        green = x_gray
        blue = tf.clip_by_value(-response*s.amp, 0.0, 1.0)
        s.img_response = tf.cast(tf.concat([red,green,blue], axis=3)*255.0, tf.uint8)

        with tf.compat.v1.name_scope("target"):
            s.y = tf.compat.v1.placeholder(tf.float32, shape=(None, s.IMAGE_HEIGHT, s.IMAGE_WIDTH), name="y")
            print(s.y)
            y = tf.reshape(s.y, (-1,s.IMAGE_HEIGHT, s.IMAGE_WIDTH,1))
            print(y)

        with tf.compat.v1.name_scope("loss"):
            absy = tf.abs(y)
            s.loss = tf.reduce_sum(input_tensor=tf.pow(response - y,2)*absy)/tf.reduce_sum(input_tensor=absy)
            s.loss += s.reg * sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

        s.global_step = tf.Variable(0, name='global_step',trainable=False)

        with tf.compat.v1.name_scope("train"):
            s.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=s.lr).minimize(s.loss, global_step=s.global_step)

        s.session = tf.compat.v1.Session()        
        s.session.run(tf.compat.v1.global_variables_initializer())
        s.saver = tf.compat.v1.train.Saver()
        
        #Start from previous checkpoint
        try:
            ckpt = s.getCKPT()
            s.saver.restore(s.session, ckpt)
            print("Model restored from %s" % (ckpt))
        except:
            print("starting from scratch")
            pass
    def getCKPT(s):
        return s.path + '/model.ckpt'
        
    
    def train(s, images, targets, lr, reg, amp):
        return s.session.run(fetches=[s.train_step, s.loss, s.img_response, s.global_step],feed_dict={s.x_uint8: images, s.y: targets, s.lr: lr, s.reg: reg, s.amp: amp})
    
    #evaluate single image
    def evaluate(s, images, amp):
        return s.session.run(fetches=[s.img_response],feed_dict={s.x_uint8: images, s.amp: amp})

    def save(s):
        p = s.getCKPT()
        print(p)
        s.saver.save(s.session, p)
