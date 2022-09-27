using MDToolbox
using Statistics
using LinearAlgebra
using BenchmarkTools
using Dates

t1 = now()
########################################

function printlnsep(args...; sep::String = " ")
    println(join(args, sep))
end

########################################

#path="/Users/yulian/MY_DATA/LU/TECH/gr/info_allostery/dex_f602s/"
path="."
pdb=ARGS[1]
dcd=ARGS[2]
paths_file=ARGS[3]

if size(ARGS,1) != 3
        println("three inputs required: pdb xtc paths_file")
end


#t = mdload("$path/cort_wt_CA_cortC9_dt1_second100ns.pdb")
#t = mdload("$path/cort_wt_CA_cortC9_dt1_100ns.dcd",top=t)

t = mdload("$path/$pdb")
t = mdload("$path/$dcd",top=t)

########################################
function read_simply_formatted_paths_file(path_to_file)

    all_paths = Vector{Vector{Int64}}()
    all_paths_length = Vector{Float64}()

    #open("$path/simply_formatted_paths_short.txt") do f
    open("$path_to_file") do f

        temp_path = Vector{Int64}() 
        lines = readlines(f) # read from file

        for i in 1:size(lines,1)
            pieces = split(lines[i], ' ', keepempty=false)
            for j in 1:size(pieces,1)  
                piece= parse(Float64, pieces[j])
                if j ==1
                    push!(all_paths_length,piece)
                else    
                    push!(temp_path,piece+1) # index of input paths is pdb_res_index-1
                end

            end
            push!(all_paths,temp_path)
            temp_path = [] 
        end
    end

    # println(all_paths)
    # println(all_paths_length)

    return all_paths, all_paths_length

end

########################################

all_paths, all_paths_length = read_simply_formatted_paths_file("$path/$paths_file")


########################################

function get_dr_for_a_block(t;step=step,block_numb=block_numb)

    coord_block = t.xyz[1+step*(block_numb-1):step*block_numb,:]
    
    mean_t = mean(coord_block, dims=1)
    
    dr = zeros(size(coord_block,1),size(coord_block,2))
    for i = 1:size(coord_block,1) # for frame
        #println(size(t.xyz[i,:]))
        dr[i,:] = coord_block[i,:] .- mean_t[1,:]
    end

    return dr

end

function get_dr(t;step=step)

    mean_t = mean(t.xyz, dims=1)
    
    dr = zeros(size(t.xyz,1),size(t.xyz,2))
    for i = 1:size(t.xyz,1) # for frame
        #println(size(t.xyz[i,:]))
        dr[i,:] = t.xyz[i,:] .- mean_t[1,:]
    end

    return dr

end


function dot_product_res_res(t,dr;res1,res2,step)

    #dot_prod_in_frames = zeros(size(t.xyz,1))
    dot_prod_in_frames = zeros(size(dr,1))

    #for frame = 1:size(t.xyz,1) # for frame
    for frame = 1:step # for frame

        dot_prod_in_frames[frame] = dot(dr[frame,3*res1-2:3*res1],dr[frame,3*res2-2:3*res2])

    end

    return dot_prod_in_frames

end
   
function dot_product_res_res_with_tau_shift(t,dr;res1,res2,tau,step)

    #dot_prod_in_frames = zeros(size(t.xyz,1)-tau)
    dot_prod_in_frames = zeros(size(dr,1)-tau)

    #dr_res1 = dr[1:size(t.xyz,1)-tau,3*res1-2:3*res1]
    #dr_res2 = dr[1+tau:size(t.xyz,1),3*res2-2:3*res2]
    dr_res1 = dr[1:size(dr,1)-tau,3*res1-2:3*res1]
    dr_res2 = dr[1+tau:size(dr,1),3*res2-2:3*res2]
    
    for frame = 1:step-tau # for frame

        dot_prod_in_frames[frame] = dot(dr_res1[frame,:],dr_res2[frame,:])

    end
    
    return dot_prod_in_frames

end

function get_time_indep_MI(dot_prod_in_frames,denom1,denom2)
    
    numerator = mean(dot_prod_in_frames)^2
    return -0.5*log(abs(1-numerator/(mean(denom1)*mean(denom2))))
    
end

########################################

function get_info_transfer(t,dr;res1,res2,tau,step)

    dot_prod_in_frames = dot_product_res_res(t,dr;res1=res1,res2=res2,step=step)
    
    #denom1, denom2 = get_denom_time_indep_MI(dr,res1,res2)
    denom1 = dot_product_res_res(t,dr,res1=res1,res2=res1,step=step)
    denom2 = dot_product_res_res(t,dr,res1=res2,res2=res2,step=step)
    
    time_indep_MI = get_time_indep_MI(dot_prod_in_frames,denom1,denom2)
    #print ("time_indep_MI: ", round(time_indep_MI,3))
    
    ############
    
    numerator1 = mean(dot_prod_in_frames)^2
    
    dot_prod_in_frames_tau_shift = dot_product_res_res_with_tau_shift(t,dr,res1=res1,res2=res2,tau=tau,step=step)
    numerator2 = mean(dot_prod_in_frames_tau_shift)^2
    
    dot_prod_in_frames_res2_res2 = dot_product_res_res(t,dr,res1=res2,res2=res2,step=step)
    numerator3 = mean(dot_prod_in_frames_res2_res2)
    # numerator3 = mean(dr[:,res2,:]^2,axis=0)
    
    numerator4 = 2*mean(dot_prod_in_frames)
    
    dot_prod_in_frames_tau_shift_res2_res2 = dot_product_res_res_with_tau_shift(t,dr,res1=res2,res2=res2,tau=tau,step=step)
    numerator5 = mean(dot_prod_in_frames_tau_shift_res2_res2)
    
    numerator6 = mean(dot_prod_in_frames_tau_shift)

    numerator_All = (numerator1 + numerator2)*numerator3 - numerator4*numerator5*numerator6
    
    ############
    
    denominator1 = numerator3^2
    denominator2 = numerator5^2
    dot_prod_in_frames_res1_res1 = dot_product_res_res(t,dr,res1=res1,res2=res1,step=step)
    denominator3 = mean(dot_prod_in_frames_res1_res1)
    
    denominator1_All = (denominator1-denominator2)*denominator3
    
    time_depended_MI = -0.5*log(abs(1-numerator_All/denominator1_All))
    #print("time_depended_MI: ", round(time_depended_MI,3))
    
    Tij = time_depended_MI-time_indep_MI
    #print ("Tij = ",round(Tij,3))
    return time_indep_MI, time_depended_MI, Tij
end

########################################

function get_dr_in_blocks(t,step,nblocks)

    dr_in_blocks = zeros(nblocks,step,Int64(size(t.xyz,2)))
    
    for block_numb in 1:nblocks # nblocks
    
        #dr = get_dr_in_blocks(t;step=step,block_numb=block_numb)
        dr_in_blocks[block_numb,:,:] = get_dr_for_a_block(t;step=step,block_numb=block_numb)
    end

    return dr_in_blocks
        
end

########################################

function get_info_transfer_per_path(t,dr_in_blocks,step,path1,nblocks,all_tau)

    avg_tij_in_pairs_in_blocks = zeros((size(path1,1)-1))
    avg_time_indep_MI_in_pairs_in_blocks = zeros((size(path1,1)-1))
    avg_time_depended_MI_in_pairs_in_blocks = zeros((size(path1,1)-1))

    avg_tij_in_pairs_all_tau = zeros(((size(path1,1)-1),size(all_tau,1)))
    avg_time_indep_MI_in_pairs_all_tau = zeros(((size(path1,1)-1),size(all_tau,1)))
    avg_time_depended_MI_in_pairs_all_tau = zeros(((size(path1,1)-1),size(all_tau,1)))

    for block_numb in 1:nblocks # nblocks

        dr = dr_in_blocks[block_numb,:,:]

        for pair in 1:(size(path1,1)-1)

            res2 = path1[pair]
            res1 = path1[pair+1]

            for tau_ndx in 1:size(all_tau,1)

                #println(all_tau[tau_ndx])
                time_indep_MI, time_depended_MI, Tij = get_info_transfer(t,dr,res1=res1,res2=res2,tau=all_tau[tau_ndx],step=step)
                #println(time_indep_MI," ", time_depended_MI," ", Tij)
                avg_tij_in_pairs_all_tau[pair,tau_ndx] += Tij
                avg_time_indep_MI_in_pairs_all_tau[pair,tau_ndx] += time_indep_MI
                avg_time_depended_MI_in_pairs_all_tau[pair,tau_ndx] += time_depended_MI

            end
        end 
    end

    #avg_tij_in_pairs_all_tau

    max_tij_in_pairs = maximum(avg_tij_in_pairs_all_tau,dims = 2) / nblocks
    tau_of_max_Tij_ndx = argmax(avg_tij_in_pairs_all_tau,dims = 2)[:]

    max_tau_in_pairs = zeros(Int64,0)
    max_indep_MI_in_pairs = zeros(0)
    max_depended_MI_in_pairs = zeros(0)

    for i in tau_of_max_Tij_ndx
        append!(max_tau_in_pairs,all_tau[convert(Tuple,i)[2]])
        append!(max_indep_MI_in_pairs,avg_time_indep_MI_in_pairs_all_tau[convert(Tuple,i)[1],convert(Tuple,i)[2]])
        append!(max_depended_MI_in_pairs,avg_time_depended_MI_in_pairs_all_tau[convert(Tuple,i)[1],convert(Tuple,i)[2]])   
    end
    max_indep_MI_in_pairs /=nblocks
    max_depended_MI_in_pairs /=nblocks

    # println(max_tau_in_pairs)
    # println(max_tij_in_pairs)
    # println(max_indep_MI_in_pairs)
    # println(max_depended_MI_in_pairs)

    path_Tij = 0
    path_Info = 0

    path_Tij_min = 10000
    path_Info_min = 10000

    for pair in 1:(size(path1,1)-1)
        path_Tij += max_tij_in_pairs[pair]
        path_Info += max_tij_in_pairs[pair]/max_tau_in_pairs[pair]

        if max_tij_in_pairs[pair] < path_Tij_min
            path_Tij_min = max_tij_in_pairs[pair]
            path_Info_min = max_tij_in_pairs[pair]/max_tau_in_pairs[pair]
        end
    end

    return path_Tij, path_Info, path_Tij_min, path_Info_min

end

########################################

all_tau=[1,2,3,4,5,6,7,8,9,10] 
step=100
nblocks=10
dr_in_blocks = get_dr_in_blocks(t,step,nblocks)

###################################
global count=1

for path1 in all_paths

    path_Tij, path_Info, path_Tij_min, path_Info_min = get_info_transfer_per_path(t,dr_in_blocks,step,path1,nblocks,all_tau)
    path_Tij_flip, path_Info_flip, path_Tij_min_flip, path_Info_min_flip = get_info_transfer_per_path(t,dr_in_blocks,step,reverse(path1),nblocks,all_tau)

    net_info_tr = path_Tij-path_Tij_flip
    rate_const = path_Info/path_Info_flip
    min_info_tr = path_Tij_min-path_Tij_min_flip

    print(all_paths_length[count], " ")
    path1=path1.-1
    
    for res in path1
    	print(res, " ")
    end

    printlnsep(path_Tij,path_Info,path_Tij_flip,path_Info_flip,net_info_tr,rate_const,path_Tij_min,path_Info_min,path_Tij_min_flip,path_Info_min_flip,min_info_tr,sep=" ")
    global count+=1
end

###################################
t2 = now()
println(canonicalize(t2 - t1))
########################################
