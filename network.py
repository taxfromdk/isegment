import tensorflow as tf
import tensorflow.contrib.slim as slim

class Network:
    def addConvLayer(s, l, features, kernel_width, dilation_rate, last):
        if last:
            l = slim.conv2d(l, features, [kernel_width, kernel_width], 1, padding='SAME', rate=dilation_rate, activation_fn=None)
        else:
            l = slim.conv2d(l, features, [kernel_width, kernel_width], 1, padding='SAME', rate=dilation_rate)    
        s.receptive_field_range += ((kernel_width-1))/2*dilation_rate
        print( l )
        return l

    def __init__(s, IMAGE_HEIGHT, IMAGE_WIDTH, path):
        print("Tensorflow version: ", tf.__version__)
        tf.set_random_seed(42)
        tf.reset_default_graph()

        s.IMAGE_HEIGHT = IMAGE_HEIGHT
        s.IMAGE_WIDTH = IMAGE_WIDTH
        s.path = path

        with tf.name_scope("input"):
            s.x_uint8 = tf.placeholder(tf.uint8, shape=(None, s.IMAGE_HEIGHT, s.IMAGE_WIDTH, 3), name="x")
            print(s.x_uint8)    
            #range 0 to 1
            x_float32 = tf.cast(s.x_uint8, tf.float32)/255.0
            #take a grayscale copy
            x_gray = tf.image.rgb_to_grayscale(x_float32)
            #move to raange -1 to 1
            x_float32 = (x_float32-0.5)*2.0
            print("x_float32",x_float32)
            print("x_gray",x_gray)

            s.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            s.reg = tf.placeholder(tf.float32, shape=(), name="regularization")
            s.amp = tf.placeholder(tf.float32, shape=(), name="regularization")


        s.receptive_field_range = 0 
            
        with tf.name_scope("convolutions"):
            
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

        with tf.name_scope("target"):
            s.y = tf.placeholder(tf.float32, shape=(None, s.IMAGE_HEIGHT, s.IMAGE_WIDTH), name="y")
            print(s.y)
            y = tf.reshape(s.y, (-1,s.IMAGE_HEIGHT, s.IMAGE_WIDTH,1))
            print(y)

        with tf.name_scope("loss"):
            absy = tf.abs(y)
            s.loss = tf.reduce_sum(tf.pow(response - y,2)*absy)/tf.reduce_sum(absy)
            s.loss += s.reg * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        s.global_step = tf.Variable(0, name='global_step',trainable=False)

        with tf.name_scope("train"):
            s.train_step = tf.train.AdamOptimizer(learning_rate=s.lr).minimize(s.loss, global_step=s.global_step)

        s.session = tf.Session()        
        s.session.run(tf.global_variables_initializer())
        s.saver = tf.train.Saver()
        
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
    