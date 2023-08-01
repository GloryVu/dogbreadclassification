import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dogbreadclassification.classifier.prune import PruneThread
from dogbreadclassification.classifier.train import TrainThread
train = TrainThread(arch = 'resnet152',batch_size=32)
print('--------')
train.start()
while(train.is_alive()):
    time.sleep(1)
prune = PruneThread(arch = 'resnet152',batch_size=32)
prune.start()
while(prune.is_alive()):
    time.sleep(1)