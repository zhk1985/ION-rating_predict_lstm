import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from collections import Counter
from keras.models import Sequential, Model
from keras.layers import GRU,LSTM,Dense,Dropout,TimeDistributed,Reshape,concatenate,Embedding,Dense, Input

import matplotlib.pyplot as plt


# define problem properties
data = pd.read_csv('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ionall.csv')
data=data.loc[((data['TCastOriginator']!='FOX') & (data['TCastOriginator']!='CW')),:]
data.loc[data['TCastOriginator']=='ABC','P2554']=data.loc[data['TCastOriginator']=='ABC','P2554']/2000000  #adjust weight of ION sample higher
data.loc[data['TCastOriginator']=='AMC','P2554']=data.loc[data['TCastOriginator']=='AMC','P2554']/650000
data.loc[data['TCastOriginator']=='CBS','P2554']=data.loc[data['TCastOriginator']=='CBS','P2554']/2500000
data.loc[data['TCastOriginator']=='CW','P2554']=data.loc[data['TCastOriginator']=='CW','P2554']/600000
data.loc[data['TCastOriginator']=='FOX','P2554']=data.loc[data['TCastOriginator']=='FOX','P2554']/1900000
data.loc[data['TCastOriginator']=='ION','P2554']=data.loc[data['TCastOriginator']=='ION','P2554']/300000
data.loc[data['TCastOriginator']=='NBC','P2554']=data.loc[data['TCastOriginator']=='NBC','P2554']/2500000
data.loc[data['TCastOriginator']=='USA NETWORK','P2554']=data.loc[data['TCastOriginator']=='USA NETWORK','P2554']/700000
data.loc[data['TCastOriginator']=='WGN AMERICA','P2554']=data.loc[data['TCastOriginator']=='WGN AMERICA','P2554']/100000


xscaler=MinMaxScaler()
xscaler.fit(data.iloc[:,4:12])
data.iloc[:,4:12]=xscaler.transform(data.iloc[:,4:12])
yscaler = MinMaxScaler()
yscaler.fit(data.iloc[:,21])
data.iloc[:,21]=yscaler.transform(data.iloc[:,21])

ct=Counter(data['BroadcastDate']+data['TCastOriginator'])
t_len =max(ct.items(), key=lambda x:x[1])[1]
sample_len = len(set(data['BroadcastDate']+data['TCastOriginator']))
x_len = 8
SHOW_DIMENSION=3  #in a multidimension space to represent a show
NET_DIMENSION=3 #in a multidimension space to represent a net

# create a sequence classification instance
shows=list(set(data['show']))
show_to_index=dict(zip(shows,np.array(range(len(shows)))+1))
index_to_show=dict(zip(np.array(range(len(shows)))+1,shows))
show_to_index['NA']=0
index_to_show[0]='NA'

nets=list(set(data['TCastOriginator']))
net_to_index=dict(zip(nets,np.array(range(len(nets)))+1))
index_to_net=dict(zip(np.array(range(len(nets)))+1,nets))
net_to_index['NA']=0
index_to_net[0]='NA'

#prepare training data
x1=np.zeros([sample_len,t_len,x_len]) #variables except show and  net
x2=np.zeros([sample_len,t_len,])    #show, will be embeddings to [batch,t_len,SHOW_DIMENSION] later
x3=np.zeros([sample_len,t_len,])
y=np.zeros([sample_len,t_len])
i=0
j=0
k=0
d_pre=data.iloc[0]
d=data.iloc[0]
while True:
  x1[j,k,:]=d[4:12]
  x2[j,k]=show_to_index[d[13]]
  x3[j,k]=net_to_index[d[14]]
  y[j,k]=d[21]
  d_pre=d
  i=i+1
  k=k+1
  if i>=data.shape[0]:
   break
  d=data.iloc[i]
  if i>0 and (d[1]!=d_pre[1] or d[14]!=d_pre[14]):
   k=0
   j=j+1


#flag=np.array(range(sample_len))<0.9*sample_len
#flag=np.random.uniform(sample_len)<0.8
flag=np.repeat(True,sample_len)
flag[7900:8033]=False    #last quater of ion
x1_train=x1[flag,:,:]
x2_train=x2[flag,:]
x3_train=x3[flag,:]
y_train=y[flag,:]
x1_test=x1[~flag,:,:]
x2_test=x2[~flag,:]
x3_test=x3[~flag,:]
y_test=y[~flag,:]

# define LSTM
x1_in=Input(shape=(t_len,x_len),dtype='float32')
x2_in=Input(shape=(t_len,),dtype='int32')
x3_in=Input(shape=(t_len,),dtype='int32')
x_show=Embedding(input_dim=len(show_to_index), output_dim=SHOW_DIMENSION, embeddings_initializer='zeros', input_length=t_len)(x2_in)
x_net=Embedding(input_dim=len(net_to_index), output_dim=NET_DIMENSION, embeddings_initializer='uniform', input_length=t_len)(x3_in)
y=concatenate([x_show,x_net,x1_in], axis=2)
y=LSTM(50, input_shape=(t_len,SHOW_DIMENSION+NET_DIMENSION+x_len), return_sequences=True)(y)
#y=GRU(30, input_shape=(t_len,SHOW_DIMENSION+NET_DIMENSION+x_len), return_sequences=True)(y)
y=Dropout(0.6)(y)
y=LSTM(50, return_sequences=True)(y)
#y=GRU(30, return_sequences=True)(y)
y=Dropout(0.6)(y)
y=TimeDistributed(Dense(1, activation='sigmoid'))(y)
y=Reshape((t_len,))(y)
model = Model(inputs=[x1_in,x2_in,x3_in], outputs=y)

opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])


# fit model for one epoch on this sequence
his=model.fit([x1_train,x2_train,x3_train], y_train, epochs=3000, batch_size=500, verbose=2,validation_data=([x1_test,x2_test,x3_test],y_test))
model.save('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ionall_2d_1000.mdl')
model.save_weights('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ionall_2d_1000.wgt')
#model.load_weights('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ionall_3000.wgt')
#model=keras.models.load_model('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ionall_3000.mdl')

#model.evaluate([x1_train,x2_train,x3_train],y_train,batch_size=50, verbose=0)
#model.evaluate([x1_test,x2_test,x3_test],y_test,batch_size=50, verbose=0)
#a=model.get_layer(index=1).get_weights()[0][:,0]
#b=['na']+shows
#c=dict(zip(b,a))
# evaluate LSTM
y_pre = yscaler.inverse_transform(model.predict([x1_test,x2_test,x3_test], verbose=2))
y_test_or= yscaler.inverse_transform(y_test)
for i in range(t_len):
	print('Expected:',y_test_or[5000, i]*300000, 'Predicted', y_pre[5000, i]*300000)
	
#predict with any possible schedule
x1_re=np.zeros([1,t_len,x_len])
x2_re=np.zeros([1,t_len,])
x3_re=np.zeros([1,t_len,])

x1_re[0,0:18,0]=np.repeat(2018,18)
x1_re[0,0:18,1]=np.repeat(3,18)
x1_re[0,0:18,2]=np.repeat(9,18)
x1_re[0,0:18,3]=np.repeat(5,18)   #weekday
x1_re[0,0:18,4]=np.array([9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2])
x1_re[0,0:18,5]=np.repeat(0,18)
x1_re[0,0:18,6]=np.repeat(60,18)
x1_re[0,0:18,7]=np.repeat(1,18)
x1_re[0,:,:]=xscaler.transform(x1_re[0,:,:])
x1_re[0,18:,:]=0
#x2_re[0,0:18]=show_to_index['CRIMINAL MINDS']
#x2_re[0,0:18]=show_to_index['HAWAII FIVE-0']
#x2_re[0,0:18]=show_to_index['CHICAGO PD']
#x2_re[0,0:18]=show_to_index['NCIS']
#x2_re[0,0:18]=show_to_index['NCIS: LOS ANGELES']
x2_re[0,0:18]=show_to_index['CSI']

x3_re[0,0:18]=net_to_index['ION']

r=yscaler.inverse_transform(model.predict([x1_re,x2_re,x3_re]))*300000
r=yscaler.inverse_transform(model.predict([x1_train[7000:7001,:,:],x2_train[7000:7001,:],x3_train[7000:7001,:]]))*300000