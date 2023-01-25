import numba as nb
import numpy as np

'''
uses high charge atoms to suggest molecule position
'''

@nb.jit(nopython=True)
def calculate_molecular_positions(hca,r):
    points = [0,0.25,0.5,0.75,1.0]
        
    x_cm = [2]
    y_cm = [2]
    z_cm = [2]
    for at in hca:
        for a_point in points:
            x = a_point - r[at,0]
            if x >= 0.0 and x <= 1.0:
                x_cm.append(x)
                
        for b_point in points:
            y = b_point - r[at,1]
            if y >= 0.0 and y <= 1.0:
                y_cm.append(y)
        
        for c_point in points:
            z = c_point - r[at,2]
            if z >= 0.0 and z <= 1.0:
                z_cm.append(z)
    
    x_cm.sort()
    y_cm.sort()
    z_cm.sort()
    
    # Cluster possible x_cm values
    n_pos = 1
    centroid = x_cm[0]
    x_cm_cen = []
    for i in range(1,len(x_cm)):
        if x_cm[i] - x_cm[i-1] > 0.02:
            x_cm_cen.append(centroid / n_pos)
            n_pos = 1
            centroid = x_cm[i]
        else:
            n_pos += 1
            centroid += x_cm[i]
            
    n_pos = 1
    centroid = y_cm[0]
    y_cm_cen = []
    for i in range(1,len(y_cm)):
        if y_cm[i] - y_cm[i-1] > 0.02:
            y_cm_cen.append(centroid / n_pos)
            n_pos = 1
            centroid = y_cm[i]
        else:
            n_pos += 1
            centroid += y_cm[i]
            
    n_pos = 1
    centroid = z_cm[0]
    z_cm_cen = []
    for i in range(1,len(z_cm)):
        if z_cm[i] - z_cm[i-1] > 0.02:
            z_cm_cen.append(centroid / n_pos)
            n_pos = 1
            centroid = z_cm[i]
        else:
            n_pos += 1
            centroid += z_cm[i]
    
    r_cm = []
    for x in x_cm_cen:
        for y in y_cm_cen:
            for z in z_cm_cen:
                if [x,y,z] not in r_cm:
                    r_cm.append([x,y,z])
    return r_cm
