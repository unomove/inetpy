from engine.pyalice import *

import os
import os.path as osp
import numpy as np
import scipy.misc

from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from net import IntentionNet
from dataset import preprocess_input
from PIL import Image

# A Python codelet for inet control
# For comparison, please see the same logic in C++ at "PingCpp.cpp".

class INETPython(Codelet):
    def start(self):
        # This part will be run once in the beginning of the program
        # We can tick periodically, on every message, or blocking. See documentation for details.
        self.rx_intention = self.isaac_proto_rx("Vector3fProto", "intention_control")
        self.rx_left = self.isaac_proto_rx("ColorCameraProto", "left")
        self.rx_mid = self.isaac_proto_rx("ColorCameraProto", "mid")
        self.rx_right = self.isaac_proto_rx("ColorCameraProto", "right")
        self.tx = self.isaac_proto_tx("StateProto", "cmd_vel")

        self.tick_on_message(self.rx_mid)

        model_fn = self.config['model_fn']
        self.mode = self.config['mode']
        self.input_frame = self.config['input_frame']
        self.num_control = self.config['num_control']
        self.num_intentions = self.config['num_intentions']
        # set keras session
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction=0.8
        KTF.set_session(tf.Session(config=config_gpu))

        # load model
        model = IntentionNet(self.mode, self.input_frame, self.num_control, self.num_intentions)
        # load checkpoint
        model.load_weights(model_fn)
        print ("=> loaded checkpoint '{}'".format(model_fn))
        self.model = model

    def predict_cmd(self, image, intention):
        if self.input_frame == 'MULTI':
            rgb = [np.expand_dims(preprocess_input(im), axis=0) for im in image]
        else:
            rgb = [np.expand_dims(preprocess_input(image), axis=0)]

        if self.mode == 'DLM':
            i_intention = to_categorical([intention], num_classes=self.num_intentions)
        else:
            i_intention = np.expand_dims(preprocess_input(intention), axis=0)

        if self.input_frame == 'NORMAL':
            pred_control = self.model.predict(rgb + [i_intention])
        elif self.input_frame == 'MULTI':
            pred_control = self.model.predict(rgb+[i_intention])

        return pred_control

    def tick(self):
        rx_intention_msg = self.rx_intention.message
        rx_left_msg = self.rx_left.message
        rx_mid_msg = self.rx_mid.message
        rx_right_msg = self.rx_right.message
   
        latest_intention=None
        latest_left=None
        latest_mid=None
        latest_right=None
        if rx_intention_msg is not None:
            latest_intention = rx_intention_msg
        if rx_left_msg is not None:
            latest_left = rx_left_msg
        if rx_mid_msg is not None:
            latest_mid = rx_mid_msg
        if rx_right_msg is not None:
            latest_right = rx_right_msg

        if latest_intention is None and latest_left is None and latest_right is None and latest_mid is None:
            return

        intention = int(latest_intention.proto.z)

        input_size = (224, 224)
        left = Image.fromarray(latest_left.tensor)
        nl = left.resize(input_size)
        mid = Image.fromarray(latest_mid.tensor)
        nm = mid.resize(input_size)
        right = image.fromarray(latest_right.tensor)
        nr = right.resize(input_size)

        cmd_vel = self.predict_cmd([left, mid, right], intention)
        print ("cmd vel", cmd_vel)

        tx_message = self.tx.init()
        data = tx_message.proto.init('data', 2)
        data[0] = cmd_vel[0]   # linear speed
        data[1] = cmd_vel[1]    # angular speed
        self.tx.publish()

def main():
    app = Application(app_filename="apps/spot/inetpy/inet.app.json")
    app.nodes["inet"].add(INETPython)
    app.connect('left_d435i.camera/realsense', 'color',
                'inet/PyCodelet', 'left')
    app.connect('mid_d435i.camera/realsense', 'color',
                'inet/PyCodelet', 'mid')
    app.connect('right_d435i.camera/realsense', 'color',
                'inet/PyCodelet', 'right')
    app.connect('inet/PyCodelet', 'cmd_vel', 'commander.subgraph/interface',
                'control')
    app.start_wait_stop()

if __name__ == '__main__':
    main()
