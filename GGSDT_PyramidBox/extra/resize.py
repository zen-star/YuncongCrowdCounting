import cv2

scale = 1.2 

with open('./GGSDT_PyramidBox/results/test_final_res_2.txt','w') as fw:
    with open('./GGSDT_PyramidBox/results/test_final_res.txt','r') as fr:
        newlines = []
        while True:
            line = fr.readline()
            if not line:
                break
            fields = line.strip().split(' ')
            if len(fields) == 1:
                if len(fields[0])>4:
                    im = cv2.imread('./yuncong_test_set/'+fields[0].replace('JPG','jpg'))
                    if im is None:
                        im = cv2.imread('./yuncong_test_set/'+fields[0].replace('jpg','JPG'))
                    assert im is not None,fields[0]
                newlines.append(line)
            else:
                line1 = line.strip().split(' ')
                x = int(line1[0])
                y = int(line1[1])
                w = int(line1[2])
                h = int(line1[3])
                s = float(line1[4])
        
                x = int(x - (scale-1)*w/2)
                y = int(y - (scale-1)*h/2)
                w = int(w*scale)
                h = int(h*scale)
                if x + w >= im.shape[1]:
                    w = im.shape[1]-1-x
                if y + h >= im.shape[0]:
                    h = im.shape[0]-1-y
                if x < 0:
                    x = 0.01
                if y < 0:
                    y = 0.01
                
                newlines.append(' '.join([str(int(x)),str(int(y)),str(int(w)),str(int(h)),str(s)[:5]])+'\n')

        for line in newlines:
            fw.write(line)
