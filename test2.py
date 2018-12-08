import pickle
import sys
sys.path.append('/work/wangzhen/Attention-LSTM')
from utilities.data_loader import get_embeddings, Task4Loader, prepare_dataset
loader = Task4Loader(None, 27)
wholeFile = "dataset/brexit/brexit5cross.txt"
training, testing = loader.load_stance_brexit_5cross(wholeFile,0)
