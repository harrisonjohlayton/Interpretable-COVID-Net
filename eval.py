from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file

'''
evaluate for all inputs in test file.
creates results.txt file detailing results for each input.
'''

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2:'COVID-19'}

def eval(sess, graph, top_percent, output_testfile, output_results, testfile, testfolder, input_tensor, output_tensor, input_size):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)
    no_missing = 0

    y_test = []
    pred = []
    for i in range(len(testfile)):
        line = testfile[i].split()
        image_name = line[1]
        correct_label = line[2]
        correct_idx = mapping[correct_label]

        #ignore missing images
        if (not os.path.isfile(os.path.join(testfolder, image_name).replace('\\','/'))):
            print("MISSING: " + os.path.join(testfolder, image_name).replace('\\','/'))
            no_missing += 1
            continue
        print(os.path.join(testfolder, line[1]).replace('\\', '/'))

        x = process_image_file(os.path.join(testfolder, image_name).replace('\\', '/'), top_percent, input_size)
        x = x.astype('float32') / 255.0

        softmax_output = np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)}))
        prediction = softmax_output.argmax(axis=1)
        prediction_idx = prediction[0]
        prediction_label = inv_mapping[prediction_idx]

        y_test.append(correct_idx)
        pred.append(prediction)

        #output the answers to files
        output_testfile.write(testfile[i])
        # output_results.write(image_name + ' ' + correct_label + ' ' + prediction_label + ' ' \n')
        output_results.write(f'{image_name} {correct_label} {prediction_label} {softmax_output[0][0]} {softmax_output[0][1]} {softmax_output[0][2]} \n')

    #close test files
    output_testfile.close()
    output_results.close()

    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    #cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    #class_acc = np.array(cm_norm.diagonal())
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))
    print(f'{no_missing} images missing from the dataset')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
    parser.add_argument('--weightspath', default='models/B', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-1545', type=str, help='Name of model ckpts')
    parser.add_argument('--testfile', default='data/lists/test_COVIDx4.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    ##output for incorrect guesses
    parser.add_argument('--output_file', default='output/eval_results/no_crop/', type=str, help='Path to output folder for incorrect examples')
    parser.add_argument('--top_percent', default=0.0, type=float, help='Percent of image to crop off the top before squaring with center crop')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname).replace('\\', '/'))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname).replace('\\', '/'))

    graph = tf.get_default_graph()

    file = open(args.testfile, 'r')
    output_testfile = open(args.output_file + 'testfile.txt', 'w')
    output_results = open(args.output_file + 'results.txt', 'w')
    testfile = file.readlines()

    eval(sess, graph, args.top_percent, output_testfile, output_results, testfile, args.testfolder, args.in_tensorname, args.out_tensorname, args.input_size)