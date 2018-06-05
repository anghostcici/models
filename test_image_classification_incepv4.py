
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import contextlib
import math
import sys
import numpy as np
import unittest
import os
import pdb
from inception_v4 import inception_v4
import time

DATA_SIZE=224

def train(net_type, use_cuda, save_dirname, is_local):
    classdim = 10
    data_shape = [3, DATA_SIZE, DATA_SIZE] #resize the image to 3*299*299
    
    def fake_reader():
        while True:
            img = np.random.rand(3,DATA_SIZE,DATA_SIZE)
            lab = np.random.randint(0,9)
            yield img,lab
    
    
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if net_type == "vgg":
        print("train vgg net")
        net = vgg16_bn_drop(images)
    elif net_type == "resnet":
        print("train resnet")
        net = resnet_cifar10(images, 32)
    elif net_type == "inception":
        print("train inception net")
    #    pdb.set_trace()
        net = inception_v4(img=images,class_dim=classdim)
    else:
        raise ValueError("%s network is not supported" % net_type)

    #predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    parallel = False
    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            img_ = pd.read_input(images)
            label_ = pd.read_input(label)
            #prediction, avg_loss, acc = net_conf(img_, label_)
            net = inception_v4(img=img_,class_dim = classdim)
            predict = net 
            cost = fluid.layers.cross_entropy(input=predict, label=label_)
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(input=predict, label=label_)
            for o in [avg_cost, acc]:
                pd.write_output(o)
        avg_cost, acc = pd()
        # get mean loss and acc through every devices.
        avg_cost = fluid.layers.mean(avg_cost)
        acc = fluid.layers.mean(acc)
    else:
        predict = net
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

    # Test program 
    #test_program = fluid.default_main_program().clone(for_test=True)

    #optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=0.001,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    BATCH_SIZE = 8
    PASS_NUM = 1
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            fake_reader, buf_size=128 * 10),
        batch_size=BATCH_SIZE)
    #test_reader = paddle.batch(
    #    paddle.dataset.cifar_resize.test10(), batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
    train_reader_iter = train_reader()
    data = train_reader_iter.next()
    feed_data = feeder.feed(data)
    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        loss = 0.0
        for pass_id in range(PASS_NUM):
            for batch_id in range(100):
                exe.run(main_program, feed = feed_data)

                if (batch_id % 10) == 0:

                    print(
                        'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}, Time {4:5.6}'.
                        format(pass_id, batch_id + 1,
                               0.0, 0.0, time.clock()))                         

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_INIT_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        # The input's dimension of conv should be 4-D or 5-D.
        # Use normilized image pixels as input data, which should be in the range [0, 1.0].
        batch_size = 1
        tensor_img = numpy.random.rand(batch_size, 3, 32, 32).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)
        print("infer results: ", results[0])


def main(net_type, use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "image_classification_" + net_type + ".inference.model"

    train(net_type, use_cuda, save_dirname, is_local)
    #infer(use_cuda, save_dirname)


class TestImageClassification(unittest.TestCase):
    def test_inceptionv4_cuda(self):
    	with self.scope_prog_guard():
            main('inception', use_cuda=True)
    #def test_vgg_cuda(self):
    #    with self.scope_prog_guard():
    #        main('vgg', use_cuda=True)

    #def test_resnet_cuda(self):
    #    with self.scope_prog_guard():
    #        main('resnet', use_cuda=True)
    #pdb.set_trace()
    #def test_inceptionv4_cpu(self):
    #    with self.scope_prog_guard():
    #        main('inception', use_cuda=False)

    #def test_vgg_cpu(self):
    #    with self.scope_prog_guard():
    #        main('vgg', use_cuda=False)

    #def test_resnet_cpu(self):
    #    with self.scope_prog_guard():
    #        main('resnet', use_cuda=False)


    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
