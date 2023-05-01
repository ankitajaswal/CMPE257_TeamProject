# CMPE257_TeamProject

#### Add trained weights for inference
- Download saved model [here](https://drive.google.com/file/d/1pJto5RxEl9a1mjCfZlYCe_UY4ej2ojn1/view?usp=sharing)
- Upzip and add content to **model** directory

#### For CPU only
Try adding this before importing tensorflow
```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
```
