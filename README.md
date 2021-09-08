this project has two phases. 
1. Training Phase
 inputs:
	i1. encoder file
	i2. decoder file
	i3. number of words including "</s>" of the longest sentence in encoder file
	i4. number of words including "</s>" of the longest sentence in decoder file
	i5. number of sentence in encoder or decoder
	i6. batch size
	i7. number of layers
 outputs:
 	o1. attention.pt file
 	o2. vocab file

2. Testing Phase
 inputs:
 	i1. vocab from training output
 	i2. attention.pt from training output
