import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from collections import Counter
from keras.models import Sequential, Model
from keras.layers import LSTM,Dense,Dropout,TimeDistributed,Reshape,concatenate,Embedding,Dense, Input


# define problem properties
data = pd.read_csv('\\\\ion.media\\files\\user\\HomeDir\\NYC\\kzhong\\Desktop\\rnn totur\\ION rating\\ion.csv')
ct=Counter(data['broadcastdate'])
t_len =max(ct.items(), key=lambda x:x[1])[1]
sample_len = len(set(data['broadcastdate']))
x_len = 7
SHOW_DIMENSION=1  #in a multidimension space to represent a show

# create a sequence classification instance
shows=list(set(data['show']))
show_to_index=dict(zip(shows,np.array(range(len(shows)))+1))
show_to_index['NA']=0
index_to_show=dict(zip(np.array(range(len(shows)))+1,shows))
index_to_show[0]='NA'


x1=np.zeros([sample_len,t_len,x_len]) #variables except show and  net
x2=np.zeros([sample_len,t_len,])    #show, will be embeddings to [batch,t_len,SHOW_DIMENSION] later

y=np.zeros([sample_len,t_len])
i=0
j=0
k=0
d_pre=data.iloc[0]
d=data.iloc[0]
while True:
  x1[j,k,:]=d[2:9]
  x2[j,k]=show_to_index[d[9]]
  y[j,k]=d[10]
  d_pre=d
  i=i+1
  k=k+1
  if i>=data.shape[0]:
   break
  d=data.iloc[i]
  if i>0 and d[0]!=d_pre[0]:
   k=0
   j=j+1

yscaler = MinMaxScaler()
yscaler.fit(y)
y_nom=yscaler.transform(y)

xscaler=MinMaxScaler()
x1_nom=x1
for i in range(x1.shape[2]):
 xscaler.fit(x1[:,:,i])
 x1_nom[:,:,i]=xscaler.transform(x1[:,:,i])

flag=np.random.random(sample_len)>0.8
x1_train=x1_nom[flag,:,:]
x2_train=x2[flag,:]
y_train=y_nom[flag,:]
x1_test=x1_nom[~flag,:,:]
x2_test=x2[~flag,:]
y_test=y_nom[~flag,:]

# define LSTM
x1=Input(shape=(t_len,x_len),dtype='float32')
x2=Input(shape=(t_len,),dtype='int32')
x=Embedding(input_dim=len(show_to_index), output_dim=SHOW_DIMENSION, embeddings_initializer='uniform', input_length=t_len)(x2)
y=concatenate([x,x1], axis=2)
y=LSTM(10, input_shape=(t_len,SHOW_DIMENSION+x_len), return_sequences=True)(y)
y=Dropout(0.6)(y)
#y=LSTM(10, return_sequences=True)(y)
#y=Dropout(0.6)(y)
y=TimeDistributed(Dense(1, activation='sigmoid'))(y)
y=Reshape((t_len,))(y)
model = Model(inputs=[x1,x2], outputs=y)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'acc'])


# fit model for one epoch on this sequence
model.fit([x1_train,x2_train], y_train, epochs=500, batch_size=100, verbose=2)

model.evaluate([x1_train,x2_train],y_train,batch_size=50, verbose=0)
model.evaluate([x1_test,x2_test],y_test,batch_size=50, verbose=0)
a=model.get_layer(index=1).get_weights()[0][:,0]
b=['na']+shows
c=dict(zip(b,a))
# evaluate LSTM
y_pre = yscaler.inverse_transform(model.predict([x1_test,x2_test], verbose=2))
y_test= yscaler.inverse_transform(y_test)
for i in range(t_len):
	print('Expected:',y_test[30, i], 'Predicted', y_pre[30, i])