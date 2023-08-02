import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
st.write('Dogbreeds Classification Training Solutions')

st.write('')
st.write('News!')
st.write('All modules is likely to working well. I\'m try to build instruction Thanks')
st.write('I\'m training other type model.')
st.write('if you wanna test, please test at your machine. intructions to implement was completed at https://github.com/GloryVu/dogbreadclassification')
st.write('')
st.write('Some thing i wanna mention:')

st.write('1.Why did I choose Pruning strategy?')

st.write('time limited.')

st.write('computing resource limited')

st.write('With my knowledge, it\'s currently more exciting than quantize, or compress weight, PCA weight matrix,...')
st.write('2.Although reduce parameter strategy is not work well now, but I think it practically will work better in the furture')
st.write('Additionally, It could be affect So I will trial this strategy with large model soon :D .')

st.write('')
st.write('Lastest result')
def get_data(path) -> pd.DataFrame:
    return pd.read_csv(path)

placeholder = st.empty()

# near real-time / live feed simulation
while True:
    train_df = get_data('classifier/checkpoints/train_log.csv')
    prune_df = get_data(('classifier/checkpoints/prune_log.csv'))
    with placeholder.container():
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
            
        if prune_df.shape[0]!=0:
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
            prune_df['mode_size'] = prune_df['mode_size']/prune_df['mode_size'].max()
            prune_df['infer_time'] = prune_df['infer_time']/prune_df['infer_time'].max()
            fig, ax = plt.subplots()
            ax.set_ylim(0.0,1.0)
            bins = [i+1 for i in range(prune_df.shape[0])]
            ax.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax.bar(bins,prune_df['mode_size'],1.0,label='model size',color='moccasin')
            ax.legend(loc='best')
            st.markdown("### Reduce parameters size Chart")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            ax2.set_xlabel('epoch')
            ax2.set_ylim(0.0,1.0)
            ax2.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax2.bar(bins,prune_df['infer_time'],1.0,label='inference time',color='lightgrey')
            ax2.legend(loc='best')
            st.markdown("### Reduce latency Chart")
            st.pyplot(fig2)
            st.markdown("### Prune Data View")
            st.dataframe(prune_df)
        prune_df = get_data(('classifier/checkpoints/prune_log_2.csv'))          
        if prune_df.shape[0]!=0:
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
            prune_df['mode_size'] = prune_df['mode_size']/prune_df['mode_size'].max()
            prune_df['infer_time'] = prune_df['infer_time']/prune_df['infer_time'].max()
            fig, ax = plt.subplots()
            ax.set_xlabel('epoch')
            ax.set_ylim(0.0,1.0)
            bins = [i+1 for i in range(prune_df.shape[0])]
            ax.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax.bar(bins,prune_df['mode_size'],1.0,label='model size',color='moccasin')
            ax.legend(loc='best')
            st.markdown("### Reduce parameters size Chart")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            ax.set_xlabel('epoch')
            ax2.set_ylim(0.0,1.0)
            ax2.plot(prune_df['epoch'], prune_df['val_accuracy'], label = "validation acc",linestyle='-.')
            ax2.bar(bins,prune_df['infer_time'],1.0,label='inference time',color='lightgrey')
            ax2.legend(loc='best')
            st.markdown("### Reduce latency Chart")
            st.pyplot(fig2)
            st.markdown("### Prune Data View")
            st.dataframe(prune_df)
            # st.write(prune_df.shape[0])
    time.sleep(10)
    plt.close()