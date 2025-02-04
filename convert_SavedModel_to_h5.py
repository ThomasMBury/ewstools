import keras
import os

len_model = 500

list_files = os.listdir(f'saved_classifiers/bury_pnas_21/len{len_model}/')
pkl_files = [f for f in list_files if f[-4:]=='.pkl']

for pkl_file in pkl_files:
    model = keras.models.load_model(f'saved_classifiers/bury_pnas_21/len{len_model}/{pkl_file}')
    #Save the model into h5 format
    model_h5_name = pkl_file[:-4]+".h5"
    model.save(f'saved_classifiers/bury_pnas_21/len{len_model}/{model_h5_name}')
    print(f'saved model as {model_h5_name}')