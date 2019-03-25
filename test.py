from model import Deeplabv3
import numpy as np

channels = 2
deeplab_model = Deeplabv3(classes=channels)
# deeplab_model.load_weights("model_irobot_synthetic.h5")
res = deeplab_model.predict(np.empty((1,512,512,3)))

print (res.shape)

print ('worked!')