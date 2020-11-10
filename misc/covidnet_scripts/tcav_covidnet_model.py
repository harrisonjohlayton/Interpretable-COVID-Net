from tcav.model import PublicImageModelWrapper
import numpy as np
import tensorflow as tf

class CovidNetWrapper(PublicImageModelWrapper):

    def __init__(self, sess, model_saved_path, labels_path):
        self.image_shape = [480, 480, 3]
        self.image_value_range = [0,1]
        endpoints = dict(
            input='input_1:0',
            logit='norm_dense_1/Softmax:0',
            prediction='norm_dense_1/Softmax:0',
        )

        self.sess = sess
        super(CovidNetWrapper, self).__init__(
            sess,
            model_saved_path,
            labels_path,
            self.image_shape,
            endpoints,
            scope='v1'
        )

        self.model_name = 'covid_net'
        self.bottlenecks_tensors = self.get_bottleneck()

        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            self.y_input = tf.placeholder(tf.int64, shape=[None])
            self.pred = tf.expand_dims(self.ends['prediction'][0], 0)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self.y_input, len(self.labels)),
                    logits=self.pred))
            
            # print('****MODEL SECTION')
            # print(self.loss)
            # print(self.pred)
            # print(self.y_input)
            # print(tf.one_hot(self.y_input, len(self.labels)))
            # print('####model section \n\n\n')
        self._make_gradient_tensors()

        '''
        candidates
        flatten_1/Reshape:0
        post_relu/...
        '''
    
    def get_bottleneck(self):
        #add bottleneck
        graph = tf.compat.v1.get_default_graph()
        bn_endpoints = {}
        for op in graph.get_operations():
            # if '' in op.name and 'Reshape' in op.name:
            # if ('conv5_block3_2_conv' in op.name) and ('EConv2D_16' in op.name) and op.name.split('/')[-1] == 'Conv2D':
            #     bn_endpoints['conv'] = op.outputs[0]
            #     break
            if ('conv5_block3_out/add' in op.name):
                bn_endpoints['out_add'] = op.outputs[0]
                break
            # if ('flatten_1/Reshape' in op.name):
            #     bn_endpoints['flat2'] = op.outputs[0]
            #     break
        return bn_endpoints

    def id_to_label(self, idx):
        if (idx == 0):
            return 'normal'
        elif (idx == 1):
            return 'pneumonia'
        elif (idx == 2):
            return 'covid'
        else:
            raise NotImplementedError
    
    def label_to_id(self, label):
        if (label == 'normal'):
            return 0
        elif (label == 'pneumonia'):
            return 1
        elif (label == 'covid'):
            return 2
        else:
            raise NotImplementedError
    
    def get_gradient(self, acts, y, bottleneck_name, example):

        """Return the gradient of the loss with respect to the bottleneck_name.

        Args:
        acts: activation of the bottleneck
        y: index of the logit layer
        bottleneck_name: name of the bottleneck to get gradient wrt.
        example: input example. Unused by default. Necessary for getting gradients
            from certain models, such as BERT.

        Returns:
        the gradient array.
        """
        return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
            self.bottlenecks_tensors[bottleneck_name]: acts,
            self.y_input: y,
            # self.ends['input']: np.expand_dims(example, axis=0)
        })