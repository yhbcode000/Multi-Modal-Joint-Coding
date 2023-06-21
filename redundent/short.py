import numpy as np

with open('data.txt') as f:
	c = f.readlines()
	c_ = c[40480:]
	C = [line.strip('\n').split('\t') for line in c_]
	print(len(C),type(C[0]),C[0])
	print(float(C[0][0]))
	k = 0
	for i in range(0,200):
		cur = []
		for j in range(i*72,(i+1)*72):
			if len(C[j][0])==18:
				#print(C[j][0],'*')
				cur.append(float(C[j][0]))
		print(np.mean(cur))
		k+=1
