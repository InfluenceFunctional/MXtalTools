import numpy as np

'''
generate a set of perpendicular vectors
not sure what for
'''

def perpendicular_vectors(n_max):
    ''' Set the possible values for the components of the vector '''
    dv = []
    for n in range(n_max,0,-1):
        dv.append(-n)
        dv.append(n)
    dv.append(0)
    
    ''' Generate initial list of vectors '''
    vecs = []
    for x in dv:
        for y in dv:
            for z in dv:
                if x == y == z == 0:
                    continue
                if x * y * z != 0:
                    continue
                
                vecs.append([x,y,z])
    
    ''' Remove duplicates '''
    vec_list = []
    for i in range(len(vecs) - 1):
        parallel = False
        for j in range(i + 1, len(vecs)):
            vec1 = vecs[i]
            vec2 = vecs[j]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            prod = abs(np.dot(vec1,vec2)) / norm1 / norm2
            if abs(prod - 1.0) < 0.00001:
                parallel = True
        if not parallel:
            if abs(vec1[0]) != n_max and abs(vec1[1]) != n_max and abs(vec1[2]) != n_max:
                continue
            vec_list.append(vec1)
            
    return vec_list