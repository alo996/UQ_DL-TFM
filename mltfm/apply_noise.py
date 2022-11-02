import numpy as np
import tables
from tqdm import tqdm

sigma=1e-3
f = tables.open_file('../ViT-TFM/data/displacements_25000.h5', 'r')

f_n = tables.open_file('displacement_noise.h5','w')
array_c = f_n.create_earray(f_n.root,'data',tables.Float64Atom(),(0,104,104,2))
f_n.close()

f_n = tables.open_file('displacement_noise.h5','a')

cov = [[sigma**2,0],[0,sigma**2]]
for i in tqdm(range(4000)):
	X = f.root.data[i*100:(i+1)*100]
	X_noise = np.random.multivariate_normal(np.array([0,0]),cov,(len(X),104,104))
	X_noise = X+X_noise
	f_n.root.data.append(X_noise)
	
f.close()
f_n.close()
