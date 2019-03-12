import os

threshold = 0.2

with open('./GGSDT_PyramidBox/results/test_final_res_2_th.txt','w') as fw:
    with open('./GGSDT_PyramidBox/results/test_final_res_2.txt','r') as fr:
        lines = fr.readlines()
        newlines = []
        headcount = 0
        for line in lines[-1::-1]:
            fields = line.strip().split(' ')
            if len(fields) > 2 and float(fields[-1]) > threshold:
                newlines.append(line)
                headcount += 1
            elif len(fields) == 1 and len(fields[0]) < 5:
                print(line)
                newlines.append(str(headcount)+'\n')
                headcount = 0
            elif len(fields[0]) > 4:
                newlines.append(line)
        
        for line in newlines[-1::-1]:
            fw.write(line)
            
        
