import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, train_df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import torchvision
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import wget
from dogbreadclassification.classifier import util
from dogbreadclassification.classifier.train import TrainThread
from dogbreadclassification.classifier.prune import PruneThread
import matplotlib.pyplot as plt
import threading
import json
## from dogbreeds.util import reorg_dog_data
st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

def get_data(path) -> pd.DataFrame:
    return pd.read_csv(path)

st.title("Real-Time Training Dashboard")
# st.write('I\'m training other model types.')
# st.write('if you wanna test, please test at your machine. Intructions to implement was completed at https://github.com/GloryVu/dogbreadclassification')
# st.write('Wanna test training on my server. contact me via:')
# st.write('Phone: 0886621947')
# st.write('facebook: https://www.facebook.com/vu.vinh.33865854')
# st.write('email: vuvinh0246@gmail.com')
# st.write('linkedin: https://www.linkedin.com/in/vinhvu0246/')
interactholder = st.empty()
# dashboard title


with interactholder:
    col1, col2, col3 = st.columns(3)

f= open("classifier/state.json")   
state_dict = json.load(f)
prune_df = get_data(('classifier/checkpoints/prune_log_1.csv'))
train_df = get_data('classifier/checkpoints/train_log_1.csv')
model_type =state_dict['arch']
batch_size=state_dict['batch_size']
epochs = state_dict['epochs']
default_pretrain=state_dict['default_pretrain']
pretrained_model_path= state_dict['pretrained_model_path']
use_pretrain= state_dict['use_pretrain']
prune=state_dict['prune']
lr=state_dict['lr']
# is_training = state_dict['is_training']
with col3:
    st.write('lastest state')
    st.write(state_dict)

with col1:
    trainnew_bt = st.button('train new')
    continue_bt = st.button('continue train')
with col2:    
    inputplaceholder = st.empty()
def get_train_process(arch = model_type,batch_size = batch_size, epochs = epochs
            ,default_pretrain=default_pretrain,pretrained_model_path=pretrained_model_path,use_pretrain = use_pretrain,start_epoch=0,lr = 0.001):
    return TrainThread(arch = model_type,batch_size = batch_size, epochs = epochs
            ,default_pretrain=default_pretrain,pretrained_model_path=pretrained_model_path,use_pretrain = use_pretrain,
            start_epoch=start_epoch,lr=lr)

def get_prune_process(arch = model_type,batch_size = batch_size, epochs = epochs
            ,default_pretrain=default_pretrain,pretrained_model_path=pretrained_model_path,use_pretrain = use_pretrain,start_epoch=0,lr =0.001):
    return PruneThread(arch = model_type,batch_size = batch_size, epochs = epochs*3
            ,default_pretrain=default_pretrain,pretrained_model_path=pretrained_model_path,use_pretrain = use_pretrain,start_epoch=start_epoch,lr =lr)

def start_pruning_session(train_process):
    while(train_process.is_alive()):
        time.sleep(5)
    torch.cuda.empty_cache()
    prune_process = get_prune_process(arch = state_dict['arch'],batch_size = state_dict['batch_size'], epochs = state_dict['epochs']
            ,default_pretrain=False,pretrained_model_path='classifier/models/trained_model.pth',use_pretrain = True,lr = state_dict['lr'])
    with col1:
        st.write('start pruning session')
    prune_process.start()
if trainnew_bt or train_df.shape[0] == 0:

    with inputplaceholder.container():
        st.write('Repare Dataset')

        datasetlist = os.listdir('datasets')
        datasetlist = [datasetlist[i]+ '/' + os.listdir('datasets/' + datasetlist[i])[0] for i in range(len(datasetlist))]
        if len(datasetlist)!=0:
            dataset_dir ='datasets/'  + st.selectbox('Select dataset',datasetlist,)
            st.write(f'seleced dataset dir: {dataset_dir}')
        else:
            dataset_dir='images/'
            st.write(f'Not have uploaded datasets. Use default dataset')
        # top-level filters
        PRETRAIN_MODELS_PATH = 'classifier/pretrained_models/'

        train_ratio = st.slider(
            'Select train ratio',
            50.0, 90.0, 80.0, 1.0)

        val_ratio = st.slider(
            'Select validation ratio',
            0.0, 100.0-train_ratio,(100.0-train_ratio)//2, 1.0)
        st.write(f'Train : Validation : Test == {train_ratio} : {val_ratio} : {100.0-train_ratio-val_ratio}')

        st.write('Training Type')
        model_type = st.selectbox(
            'Select model type',
            [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "resnext101_64x4d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ], 3)
        st.write('You selected:', model_type)
        batch_size = st.select_slider(
            'Select batch size',
            options=[1, 2, 4, 8, 16, 24, 32, 64, 128])
        st.write('Batch size: ', batch_size)
        epochs = st.slider(
            'Num of epochs',
            5, 100, 40, 1)
        st.write('Batch size: ', epochs)
        lr = st.slider(
            'Learing rate',
            0.001, 0.1, 0.001, 0.001, format='%.3f')
        st.write('Learning rate: ', lr)
        use_pretrain = st.checkbox('Use pre-trained model',True)
        pretrained_model_path =''
        if use_pretrain:
            default_pretrain = st.radio(
            "use default weights from Pytorch",
            ('Yes', 'No'))
            if default_pretrain == 'No':
                if not os.path.exists(PRETRAIN_MODELS_PATH+ model_type):
                        os.makedirs(PRETRAIN_MODELS_PATH+ model_type)
                tmp_path=None
                st.write('Choose Pretrained Model')
                url = st.text_input('from url:', '')
                if st.button('download'):
                    try:
                        upload_time = time.strftime("_%H_%M_%S", time.localtime())
                        file_name = 'url_' + model_type+ str(upload_time) +'.pth' 
                        response = wget.download(url, PRETRAIN_MODELS_PATH+ model_type+'/'+file_name)
                        tmp_path = PRETRAIN_MODELS_PATH+ model_type+'/'+file_name
                    except Exception as e:
                        st.write('url is incorrect')
                uploaded_file = st.file_uploader("Upload a pretrain file", accept_multiple_files=False)
                if uploaded_file is not None:
                    upload_time = time.strftime("%H_%M_%S_", time.localtime())
                    bytes_data = uploaded_file.read()
                    file_name ='upload_'+str(upload_time)+uploaded_file.name
                    st.write("filename:", file_name)
                    
                    with open(PRETRAIN_MODELS_PATH+ model_type+'/'+file_name, 'wb') as f: 
                        f.write(bytes_data)
                    # st.write(bytes_data)
                    tmp_path = PRETRAIN_MODELS_PATH+ model_type+'/'+file_name
                if tmp_path is not None:
                    try:
                        resnet = getattr(torchvision.models, model_type)
                        model = resnet(pretrained=False)
                        model.load_state_dict(torch.load(tmp_path))
                        st.write(' upload pretrained model file is correct')
                        pretrained_model_path=tmp_path
                    except Exception as e:
                        os.remove(tmp_path)
                        st.write('file upload is incorrect')
                st.write("Select pre_trained model")
                
                selectbox_pretrained_model = st.selectbox('uploaded pre_trained model',os.listdir(PRETRAIN_MODELS_PATH+model_type))
                if selectbox_pretrained_model is not None:
                    pretrained_model_path =PRETRAIN_MODELS_PATH+model_type+selectbox_pretrained_model
                    st.write("pretrained_model: "+ selectbox_pretrained_model)

        prune = st.checkbox('Apply pruning',True)
        if st.button('train'):
            
            # with col1:
            #     st.write('preparing dataset')
            st.info('Preparing dataset', icon="ℹ️")
            util.reorg_dog_data('images/',val_ratio/100,(100.0-train_ratio-val_ratio)/100)
            # with col1:
            #     st.write('finish preparing dataset.')
            st.info('Finish preparing dataset.', icon="ℹ️")
            state_dict = {'arch': model_type,
                'batch_size': batch_size, 
                'epochs':epochs ,
                'default_pretrain': default_pretrain,
                'pretrained_model_path':pretrained_model_path,
                'use_pretrain': use_pretrain,
                'prune': prune,
                'lr': lr}
            train_process = get_train_process(arch = state_dict['arch'],batch_size = state_dict['batch_size'], epochs = state_dict['epochs']
            ,default_pretrain=state_dict['default_pretrain'],pretrained_model_path=state_dict['pretrained_model_path'],use_pretrain = state_dict['use_pretrain'],
            start_epoch=0,lr = state_dict['lr'])

            with open("classifier/state.json", "w") as outfile:
                json.dump(state_dict, outfile)
            # with col1:
            #     st.write('Start training session.')
            st.info('Start training session.', icon="ℹ️")
            interactholder.empty()
            train_process.start()

            threading.Thread(target=start_pruning_session, args=(train_process,)).start()

# @st.cache_data


if continue_bt:
    if(train_df.shape[0] < state_dict['epochs']):
        train_process = get_train_process(arch = state_dict['arch'],batch_size = state_dict['batch_size'], epochs = state_dict['epochs']
            ,default_pretrain=state_dict['default_pretrain'],pretrained_model_path=state_dict['pretrained_model_path'],use_pretrain = state_dict['use_pretrain'],
            start_epoch=train_df.shape[0],lr = state_dict['lr'])
        st.info(f'Continue train from epoch {train_df.shape[0]}.', icon="ℹ️")
        interactholder.empty()
        train_process.start()
        threading.Thread(target=start_pruning_session, args=(train_process,)).start()
    elif(prune_df.shape[0] < state_dict['epochs']*3):
        
        prune_process = get_prune_process(arch = state_dict['arch'],batch_size = state_dict['batch_size'], epochs = state_dict['epochs']
            ,default_pretrain=False,pretrained_model_path='classifier/models/trained_model.pth',use_pretrain = True, start_epoch=prune_df.shape[0],lr = state_dict['lr'])
        st.info(f'Continue prune from epoch {prune_df.shape[0]}.', icon="ℹ️")
        interactholder.empty()
        prune_process.start()
    else:
        # with col1:
        #     st.write('train is final')
        st.warn('Previous training session is done.', icon="⚠️")
# creating a single-element container
# if is_training:
#     interactholder.empty()
placeholder = st.empty()

# near real-time / live feed simulation

while True:
    train_df = get_data('classifier/checkpoints/train_log_1.csv')
    prune_df = get_data(('classifier/checkpoints/prune_log_1.csv'))
    # if( )
    with placeholder.container():
        st.title('Training Process')
        st.write(time.strftime("%H:%M:%S", time.localtime()))
        if train_df.shape[0]!=0:
            train_df = train_df.sort_values(by=['epoch'])
    #    ,epoch,train_accuracy,val_accuracy,train_loss,val_loss,mode_size,infer_time

        # train_df['']
        

            # create three columns
            
            # create two columns for charts
            # fig_col1, fig_col2 = st.columns(2)
            # with fig_col1:
            fig, ax = plt.subplots()
            ax.set_xlabel('epoch')
            ax.set_ylim(0.0,2.0)
            ax.plot(train_df['epoch'], train_df['train_loss'], label = "train loss")
            ax.plot(train_df['epoch'], train_df['val_loss'], label = "validation loss",linestyle='--')
            ax.plot(train_df['epoch'], train_df['train_accuracy'], label = "train acc",linestyle='-')
            ax.plot(train_df['epoch'], train_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax.legend(loc='best')
            st.markdown("### Train Chart")
            st.pyplot(fig)


            st.markdown("### Train Data View")
            st.dataframe(train_df)
        else:
            st.write('please wait, progress will be plot below after epoch 1 finish')
        if prune and prune_df.shape[0]!=0:
            prune_df = prune_df.sort_values(by=['epoch'])
            fig, ax = plt.subplots()
            ax.set_xlabel('epoch')
            ax.set_ylim(0.0,2.0)
            ax.plot(prune_df['epoch'], prune_df['train_loss'], label = "train loss")
            ax.plot(prune_df['epoch'], prune_df['val_loss'], label = "validation loss",linestyle='--')
            ax.plot(prune_df['epoch'], prune_df['train_accuracy'], label = "train acc",linestyle='-')
            ax.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax.legend(loc='best')
            st.markdown("### Prune Chart")
            st.pyplot(fig)
            prune_df['mode_size'] = prune_df['mode_size']/train_df['mode_size'].max()
            prune_df['infer_time'] = prune_df['infer_time']/train_df['infer_time'].max()
            fig, ax = plt.subplots()
            
            ax.set_xlabel('epoch')
            ax.set_ylim(0.0,1.0)
            bins = [i for i in range(prune_df.shape[0])]
            ax.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax.bar(bins,prune_df['mode_size'],1.0,label='model size %',color='moccasin')
            ax.legend(loc='best')
            st.markdown("### Reduce parameters size Chart")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            ax2.set_xlabel('epoch')
            ax2.set_ylim(0.0,1.0)
            ax2.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax2.bar(bins,prune_df['infer_time'],1.0,label='inference time %',color='lightgrey')
            ax2.legend(loc='best')
            st.markdown("### Reduce latency Chart")
            st.pyplot(fig2)
            st.markdown("### Prune Data View")
            st.dataframe(prune_df)
            # st.write(prune_df.shape[0])
            # plt.close()
    time.sleep(10)
    plt.close()