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
from dataset import PioneerDataset as Dataset
import cv2

# A Python codelet for inet control
# For comparison, please see the same logic in C++.

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
        self.use_shared_side_model = self.config["use_shared_side_model"]
        self.auto = self.config["auto"]
        # set keras session
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction=0.8
        KTF.set_session(tf.Session(config=config_gpu))

        if self.auto:
            # load model
            model = IntentionNet(self.mode, self.input_frame, self.num_control, self.num_intentions, self.use_shared_side_model)
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

        return pred_control[0]

    def tick(self):
        if not self.auto:
            return

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

        if latest_left is None or latest_right is None or latest_mid is None:
            print ('realsense messages are not there')
            return

        if latest_intention is None:
            # default forward
            intention = 0 
        else:
            intention = int(latest_intention.proto.z)

        input_size = (224, 224)
        left = cv2.resize(latest_left.tensor, input_size)
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        mid = cv2.resize(latest_mid.tensor, input_size)
        mid = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
        right = cv2.resize(latest_right.tensor, input_size)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        # cmd_vel = self.predict_cmd([left, mid, right], intention)
        cmd_vel = self.predict_cmd([right, mid, left], intention)

        tx_message = self.tx.init()
        data = tx_message.proto.init('data', 2)
        data[0] = float(cmd_vel[0]*Dataset.SCALE_VEL)   # linear speed
        data[1] = float(cmd_vel[1]*Dataset.SCALE_STEER)   # angular speed
        # print ("cmd vel", data)
        self.tx.publish()

def main():
    app = Application(app_filename="apps/spot/inetpy/inet.app.json")
    app.nodes["inet"].add(INETPython)
    app.connect('left_d435i.camera/realsense', 'color_raw',
                'inet/PyCodelet', 'left')
    app.connect('mid_d435i.camera/realsense', 'color_raw',
                'inet/PyCodelet', 'mid')
    app.connect('right_d435i.camera/realsense', 'color_raw',
                'inet/PyCodelet', 'right')
    app.connect('inet/PyCodelet', 'cmd_vel', 'commander.subgraph/interface',
                'control')
    app.start_wait_stop()

if __name__ == '__main__':
    main()
