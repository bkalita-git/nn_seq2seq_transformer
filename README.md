this project has two phases. 
1. Training Phase
 <br>Inputs:<br>
  i1. encoder file<br>
  i2. decoder file<br>
  i3. number of words including '\</s>' of the longest sentence in encoder file<br>
  i4. number of words including '\</s>' of the longest sentence in decoder file<br>
  i5. number of sentence in encoder or decoder<br>
  i6. batch size<br>
  i7. number of layers<br>
 <br>outputs:<br>
  o1. attention.pt file<br>
  o2. vocab file<br>

2. Testing Phase
 <br>inputs:<br>
  i1. vocab from training output<br>
  i2. attention.pt from training output<br>
