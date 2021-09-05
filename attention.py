import pickle
vocab = open("vocab","rb")
vocab = pickle.load(vocab)
d_vocab = {vocab["d_vocab"][i]:i for i in vocab["d_vocab"]}
num_layers = vocab["num_layers"]
import torch

def get_word(x):
	v,i=torch.max(x,0)
	return d_vocab[i.item()+1]
	
def get_tensor(x):
	return torch.LongTensor([vocab["d_vocab"][x]])

class Attention(torch.nn.Module):
	def __init__(self,embedding):
		super(Attention,self).__init__()
		self.embedding	= embedding
		self.emb1	= torch.nn.Embedding(len(vocab["e_vocab"])+1 , self.embedding)
		self.lstm1	= torch.nn.LSTM(num_layers = num_layers,input_size=self.embedding,hidden_size=self.embedding,batch_first=True)
		self.lin1	= torch.nn.Linear(vocab["e_max_len"],vocab["e_max_len"])
		self.soft	= torch.nn.Softmax(dim=1)

		self.emb2	= torch.nn.Embedding(len(vocab["d_vocab"])+1,self.embedding,padding_idx=0)
		self.lin2	= torch.nn.Linear(vocab["e_max_len"]+self.embedding,self.embedding)
		self.lstm2	= torch.nn.LSTM(num_layers = num_layers,input_size=self.embedding,hidden_size=self.embedding,batch_first=True)
		self.lin3	= torch.nn.Linear(self.embedding,len(vocab["d_vocab"]))
		
	def forward(self,e,d):
		pe=torch.nn.utils.rnn.pack_padded_sequence(e,[len(e[0])],batch_first=True)
		e = self.emb1(pe[0])
		#print(e)
		pe=torch.nn.utils.rnn.PackedSequence(e,pe[1])
		epo,(last_hid,last_state) = self.lstm1(pe)
		
		upo = torch.nn.utils.rnn.pad_packed_sequence(epo,batch_first=True,total_length=vocab["e_max_len"])[0]
		#print(upo)
		#print(last_hid)
		word = ""
		#++
		#d = d[:,None]
		#++
		#de = self.emb2(d)
		while word!="</s>":
		#------------------------------------------------------------------------------------------#
			#modified hidden for multi_layer purpose+++++++++
			last_hid_last = last_hid[-1][None]
			#last_hid_last = last_hid
			
			c = (last_hid_last[-1,:,None]*upo).sum(dim=2)
			c = self.lin1(c)
			#print(c)
			c = self.soft(c)
			#print(c)
			c = (c[:,:,None]*upo).sum(dim=2)
			#print(c)
			#--
			d = d[:,None]
			#print(d)
			#print(l)
			#--
			de = self.emb2(d)
			#print(de)
			c = c[:,None]
			de = torch.cat([c,de],dim=2)
			#print(de)
			de = self.lin2(de)
			#print(de)
			dout,(last_hid,last_state) = self.lstm2(de,(last_hid,last_state))
			#print(dout)
			#print(last_hid)
			#break
			if len(dout)>0:
				dout = self.lin3(dout)
				dout = dout[:,-1,:]
				word = get_word(dout[0])
				print(word)
				d    = get_tensor(word)
				#++d     = last_hid
		return	
		#no problem-----------------------------------------------------------------------------------#
		
	


model = Attention(vocab["embedding"])
model.load_state_dict(torch.load('attention.pt'),strict=False)

sentence = "life </s>"
indexing = torch.LongTensor([[vocab["e_vocab"][s] for s in sentence.split()]])
d = model(indexing,get_tensor("<s>"))
