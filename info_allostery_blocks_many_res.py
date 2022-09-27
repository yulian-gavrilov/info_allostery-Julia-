import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
import time

#print (len(sys.argv), "\n");
#print (sys.argv,"\n");

start_time = time.time()

if len(sys.argv) == 2:
    system =  sys.argv[1]
else:
    print('You need to include a system name');
    print('Example: python /home/yulian/bin/info_allostery_blocks_many_res.py dibC_f602s_CA_dibC_C10')

    exit(1)


def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )

#system = "dex_f602s"
# shortest paths 2.287012750586129 264 36 222 226 227
##path1 = [264, 36, 222, 226, 227]
#path1 = [263, 35, 221, 225, 226]

wisp_out  = pd.read_csv(f"./simply_formatted_paths.txt",sep='\s+', names=[x for x in range(0,21)])
all_paths = wisp_out.iloc[:, 0::].fillna("0").astype(float) 


pdb_file=f"{system}_dt1_second100ns.pdb"
xtc_file=f"{system}_dt1_100ns.xtc"
#dex_f602s_CA_dexC9_dt1_10ns_from100ns.xtc"
#outf="info_allostery_out"
#mkdir(outf)


u1 = mda.Universe(f'{pdb_file}', f'{xtc_file}')
step=10000 # 10000 ps
nblocks=10

########################

def get_r_avg(u1,step,block_numb):
    r_avg = np.zeros_like(u1.trajectory[0].positions)

    for i in range(step):
        r_avg += u1.trajectory[block_numb*step+i].positions
    r_avg /= step

    return r_avg

def get_dr(u1,r_avg,step,block_numb):

    #dr = np.zeros_like(u1.trajectory[0].positions)
    dr = np.zeros((0,u1.trajectory[0].positions.shape[0],3), float)

    for i in range(step):
        dr_current = u1.trajectory[block_numb*step+i].positions-r_avg
        dr = np.append(dr,[dr_current],axis = 0)

    return dr

def dot_product_res_res(u1,dr,res1,res2,step):
    dot_prod_in_frames = np.empty(0)
    for frame in range(step):
        dot_prod_in_frames = np.append(dot_prod_in_frames,np.dot(dr[frame,res1],dr[frame,res2]))
    return dot_prod_in_frames

def get_denom_time_indep_MI(dr,res1,res2):
    denom1 = np.mean(dr[:,res1,:]**2,axis=0)
    denom2 = np.mean(dr[:,res2,:]**2,axis=0)
    return denom1, denom2

def dot_product_res_res_with_tau_shift(u1,dr,res1,res2,tau,step):

    dot_prod_in_frames = np.empty(0)
    dr_res1 = dr[0:dr.shape[0]-tau,res1]
    dr_res2 = dr[tau:dr.shape[0],res2]

    for frame in range(step-tau):
        dot_prod_in_frames = np.append(dot_prod_in_frames,np.dot(dr_res1[frame],dr_res2[frame]))
    return dot_prod_in_frames

def get_time_indep_MI(dot_prod_in_frames,denom1,denom2):
    numerator = np.mean(dot_prod_in_frames)**2
    #return -0.5*np.log(np.abs(1-numerator/np.dot(denom1,denom2)))
    return -0.5*np.log(np.abs(1-numerator/(np.mean(denom1)*np.mean(denom2))))


########################

def get_info_transfer(u1,dr,res1,res2,tau,step):

    dot_prod_in_frames = dot_product_res_res(u1,dr,res1,res2,step)
    
    #denom1, denom2 = get_denom_time_indep_MI(dr,res1,res2)
    denom1 = dot_product_res_res(u1,dr,res1,res1,step)
    denom2 = dot_product_res_res(u1,dr,res2,res2,step)
    
    time_indep_MI = get_time_indep_MI(dot_prod_in_frames,denom1,denom2)
    #print ("time_indep_MI: ", round(time_indep_MI,3))
    
    ############
    
    numerator1 = np.mean(dot_prod_in_frames)**2
    
    dot_prod_in_frames_tau_shift = dot_product_res_res_with_tau_shift(u1,dr,res1,res2,tau,step)
    numerator2 = np.mean(dot_prod_in_frames_tau_shift)**2
    
    dot_prod_in_frames_res2_res2 = dot_product_res_res(u1,dr,res2,res2,step)
    numerator3 = np.mean(dot_prod_in_frames_res2_res2)
    # numerator3 = np.mean(dr[:,res2,:]**2,axis=0)
    
    numerator4 = 2*np.mean(dot_prod_in_frames)
    
    dot_prod_in_frames_tau_shift_res2_res2 = dot_product_res_res_with_tau_shift(u1,dr,res2,res2,tau,step)
    numerator5 = np.mean(dot_prod_in_frames_tau_shift_res2_res2)
    
    numerator6 = np.mean(dot_prod_in_frames_tau_shift)

    numerator_All = (numerator1 + numerator2)*numerator3 - numerator4*numerator5*numerator6
    
    ############
    
    denominator1 = numerator3**2
    denominator2 = numerator5**2
    dot_prod_in_frames_res1_res1 = dot_product_res_res(u1,dr,res1,res1,step)
    denominator3 = np.mean(dot_prod_in_frames_res1_res1)
    
    denominator1_All = (denominator1-denominator2)*denominator3
    
    time_depended_MI = -0.5*np.log(np.abs(1-numerator_All/denominator1_All))
    #print("time_depended_MI: ", round(time_depended_MI,3))
    
    Tij = time_depended_MI-time_indep_MI
    #print ("Tij = ",round(Tij,3))
    return time_indep_MI, time_depended_MI, Tij


########################

def get_dr_in_blocks(u1,step,nblocks):

    dr_in_blocks = np.zeros((0,step,u1.trajectory[0].positions.shape[0],3), float)
    
    for block_numb in range(nblocks): # nblocks
    
        r_avg = get_r_avg(u1,step,block_numb=block_numb)
        dr = get_dr(u1,r_avg,step,block_numb=block_numb)
        #print(dr.shape)
        dr_in_blocks = np.append(dr_in_blocks,[dr],axis=0)
        #print(f"get_dr for block {block_numb} is ready")
        
    return dr_in_blocks

def get_info_transfer_per_path(u1,dr_in_blocks,step,path1,nblocks,all_tau):

    avg_tij_in_pairs_in_blocks = np.zeros((len(path1)-1), float)
    avg_time_indep_MI_in_pairs_in_blocks = np.zeros((len(path1)-1), float)
    avg_time_depended_MI_in_pairs_in_blocks = np.zeros((len(path1)-1), float)
    #top_tau_in_pairs = np.zeros((len(path1)-1), float)
    
    avg_tij_in_pairs_all_tau = np.zeros(((len(path1)-1),len(all_tau)), float)
    avg_time_indep_MI_in_pairs_all_tau = np.zeros(((len(path1)-1),len(all_tau)), float)
    avg_time_depended_MI_in_pairs_all_tau = np.zeros(((len(path1)-1),len(all_tau)), float)
    
    for block_numb in range(nblocks): # nblocks
    
        #r_avg = get_r_avg(u1,step,block_numb=block_numb)
        #dr = get_dr(u1,r_avg,step,block_numb=block_numb)
        dr = dr_in_blocks[block_numb]
        
        for pair in range(len(path1)-1):
    
            res2 = path1[pair]
            res1 = path1[pair+1]
    
            for tau_ndx in range(len(all_tau)):
                
                time_indep_MI, time_depended_MI, Tij = get_info_transfer(u1,dr,res1=res1,res2=res2,tau=all_tau[tau_ndx],step=step)
                
                avg_tij_in_pairs_all_tau[pair][tau_ndx] += Tij
                avg_time_indep_MI_in_pairs_all_tau[pair][tau_ndx] += time_indep_MI
                avg_time_depended_MI_in_pairs_all_tau[pair][tau_ndx] += time_depended_MI
                    
        #print ("finished block ", block_numb)
    
    max_tij_in_pairs = np.max(avg_tij_in_pairs_all_tau,axis=1) / nblocks # real avg!
    tau_of_max_Tij_ndx = np.argmax(avg_tij_in_pairs_all_tau,axis=1) 
    max_indep_MI_in_pairs = np.diagonal(np.take(avg_time_indep_MI_in_pairs_all_tau, tau_of_max_Tij_ndx,axis=1)) / nblocks
    max_depended_MI_in_pairs = np.diagonal(np.take(avg_time_depended_MI_in_pairs_all_tau, tau_of_max_Tij_ndx,axis=1)) / nblocks
    max_tau_in_pairs = np.take(all_tau,tau_of_max_Tij_ndx)

    path_Tij = 0
    path_Info = 0
    
    path_Tij_min = 10000
    path_Info_min = 10000

    for pair in range(len(path1)-1):
        path_Tij += max_tij_in_pairs[pair]
        path_Info += max_tij_in_pairs[pair]/max_tau_in_pairs[pair]
        
        if max_tij_in_pairs[pair] < path_Tij_min:
            path_Tij_min = max_tij_in_pairs[pair]
            path_Info_min = max_tij_in_pairs[pair]/max_tau_in_pairs[pair]



    #return max_tij_in_pairs, tau_of_max_Tij_ndx, max_tau_in_pairs
    return path_Tij, path_Info, path_Tij_min, path_Info_min
    
    
##########################

#all_tau=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
#             40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,300,400,500]


all_tau=[1,2,3,4,5,6,7,8,9,10] 

dr_in_blocks = get_dr_in_blocks(u1,step,nblocks)
#print(dr_in_blocks.shape)
#print(dr_in_blocks[0])


for i in range(0,all_paths.shape[0]):
#for i in range(0,5000):
    
    path1 = all_paths.iloc[i].values
    path1_corr_dist = path1[0]
    path1 = np.trim_zeros(path1)[1:].astype(int)
    
    #print("path (index from 0): ", path1,path1_corr_dist)

    path_Tij, path_Info, path_Tij_min, path_Info_min = get_info_transfer_per_path(u1,dr_in_blocks,step,path1,nblocks,all_tau)
    path_Tij_flip, path_Info_flip, path_Tij_min_flip, path_Info_min_flip = get_info_transfer_per_path(u1,dr_in_blocks,step,np.flip(path1),nblocks,all_tau)

    net_info_tr = path_Tij-path_Tij_flip
    rate_const = path_Info/path_Info_flip
    min_info_tr = path_Tij_min-path_Tij_min_flip 

    #print ("path_corr_dist","path res","path_Tij(bits)","path_Info(bits/ps)", "path_Tij_flip(bits)",
    #       "path_Info_flip(bits/ps)","net_info_transfer(bits)","rate_const")
    print(path1_corr_dist, end = " ")
    for i in path1:
        print (i, end = " ")
    print (path_Tij,path_Info,path_Tij_flip,path_Info_flip,net_info_tr,rate_const,path_Tij_min,path_Info_min,path_Tij_min_flip,path_Info_min_flip,min_info_tr,sep=" ")


print("--- %s seconds ---" % (time.time() - start_time))

## to do:
'''
1. Up to 50 paths;
2. comments out print ("finished block ", block_numb)
3. print the output to file instead
4. comment out print("path (index from 0): ", path1,path1_corr_dist)
'''



