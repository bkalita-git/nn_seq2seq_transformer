import torch
import loader1
import pickle
embedding_size = 10
encoder_file = "encoder"
decoder_file = "decoder"
encoder_max_len = 6
decoder_max_len = 7
num_lines = 6
batch_size = 3
num_layers = 4
ld = loader1.load(encoder_file,encoder_max_len,decoder_file,decoder_max_len,num_lines,batch_size)
class Attention(torch.nn.Module):
	def __init__(self,embedding):
		super(Attention,self).__init__()
		self.embedding	= embedding
		self.emb1	= torch.nn.Embedding(len(ld.e_data.vocab)+1 , self.embedding)
		self.lstm1	= torch.nn.LSTM(num_layers=num_layers,input_size=self.embedding,hidden_size=self.embedding,batch_first=True)
		self.lin1	= torch.nn.Linear(ld.e_max_len,ld.e_max_len)
		self.soft	= torch.nn.Softmax(dim=1)

		self.emb2	= torch.nn.Embedding(len(ld.d_data.vocab)+1,self.embedding,padding_idx=0)
		self.lin2	= torch.nn.Linear(ld.e_max_len+self.embedding,self.embedding)
		self.lstm2	= torch.nn.LSTM(num_layers=num_layers,input_size=self.embedding,hidden_size=self.embedding,batch_first=True)
		self.lin3	= torch.nn.Linear(self.embedding,len(ld.d_data.vocab))
		
	def forward(self,e,d):
		loss = 0
		#print(d[0])
		dx = d[0][:,:-1]
		#print(dx)
		dy = d[0][:,1:]
		#print(dy)
		dx = dx.t()
		#print(dx)
		dy = dy.t()
		#print(dy)
		
		pe = torch.nn.utils.rnn.pack_padded_sequence(e[0],e[1],batch_first=True)
		#print(pe)
		#print(e)
		epe = self.emb1(pe[0])
		#print(epe)
		epe = torch.nn.utils.rnn.PackedSequence(epe,pe[1])
		#print(epe)
		epo,(last_hid,last_state) = self.lstm1(epe)
		
		upo = torch.nn.utils.rnn.pad_packed_sequence(epo,batch_first=True,total_length=ld.e_max_len)[0]
		#print(upo)
		#print(last_hid)
		#------------------------------------------------------------------------------------------#
		for d,l in zip(dx,dy):
			#modified hidden for multi_layer purpose+++++++++
			last_hid_last = last_hid[-1][None]
			#last_hid_last = last_hid
		
			c = (last_hid_last[-1,:,None]*upo).sum(dim=2)
			#print(c)
			c = self.lin1(c)
			#print(c)
			c = self.soft(c)
			#print(c)
			c = (c[:,:,None]*upo).sum(dim=2)
			#print(c)
			d = d[:,None]
			#print(d)
			#print(l)
			de = self.emb2(d)
			#print(de)
			c = c[:,None]
			#print(c)
			de = torch.cat([c,de],dim=2)
			#print(de)
			de = self.lin2(de)
			#print(de)
			dout,(last_hid,last_state) = self.lstm2(de,(last_hid,last_state))
			
			#print(dout)
			#print(last_hid)
			#print(dout)
			non_zero_l_pos = l.nonzero().view(-1)
			l = l[non_zero_l_pos]
			dout = dout[non_zero_l_pos]
			if len(dout)>0:
				dout = self.lin3(dout)
				#print(dout)
				dout = dout[:,-1,:]
				l    = l-1
				#print(dout)
				#print(l)
				loss += error(dout,l)
			else:
				continue
		return loss
		#no problem-----------------------------------------------------------------------------------#
		
	


attention = Attention(embedding_size)
error	  = torch.nn.CrossEntropyLoss()
optim	  = torch.optim.Adam(attention.parameters(),lr=0.01)
for epoch in range(2000):
	for e,d in zip(ld.e,ld.d):
		optim.zero_grad()
		loss = attention(e,d)
		loss.backward()
		optim.step()
		#print(loss)
print(loss)
torch.save(attention.state_dict(),'attention.pt')
vocab = {"e_vocab":ld.e_data.vocab,"d_vocab":ld.d_data.vocab,"embedding":10,"e_max_len":ld.e_max_len,"d_max_len":ld.d_max_len,"num_layers":num_layers}
f = open("vocab","wb")
pickle.dump(vocab,f,pickle.HIGHEST_PROTOCOL)
