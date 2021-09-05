#import class torch
import torch
#import class Dataset and DataLoader
from torch.utils.data import Dataset,DataLoader
#import numpy
import numpy as np
class load:
	def __init__(self,encoder,e_max_len,decoder,d_max_len,num_lines,batch_size):
		self.e_max_len = e_max_len
		self.d_max_len = d_max_len
		self.e_data = loader(encoder,e_max_len,num_lines)
		self.d_data = loader(decoder,d_max_len,num_lines)
	
		self.e	= DataLoader(dataset=self.e_data,batch_size=batch_size)
		self.d	= DataLoader(dataset=self.d_data,batch_size=batch_size)

class loader(Dataset):
	def __init__(self,file_name,max_length,num_lines):
		self.num_lines	= num_lines
		self.max_length	= max_length
		self.file	= open(file_name).read()
		self.vocab	= { w:i+1 for i,w in enumerate(set(self.file.split()))}
		self.sentences  = self.file.split('\n')
		self.data	= np.zeros([num_lines,max_length])
		self.seq_length	= [len(s.split()) for s in self.sentences if len(s)>0]
		i = 0
		for s in self.sentences:
			if len(s)>0:
				for j,w in enumerate(s.split()):
					self.data[i][j] = self.vocab[w]
				i += 1
		self.data	= torch.from_numpy(self.data).long()
	def __getitem__(self,index):
		return self.data[index],self.seq_length[index]
	def __len__(self):
		return self.num_lines

