import os
import sys

rjust_num = 4

class DiffChecker:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2
    
    def check(self):
        file1 = open(self.path1, 'r', encoding='utf-8')
        file2 = open(self.path2, 'r', encoding='utf-8')
        lines1 = file1.readlines()
        lines1 = list(map(lambda x: x.strip(), lines1))
        lines2 = file2.readlines()
        lines2 = list(map(lambda x: x.strip(), lines2))

        l1, l2 = [], []
        for i, line in enumerate(lines1, start=1):
            l1.append([i, line])

        for i, line in enumerate(lines2, start=1):
            l2.append([i, line])

        file1.close()
        file2.close()

        patch1, patch2, patch = [], [], []
        len1, len2 = 0, 0
        lines2_copy = lines2[:]
        for line in l1:
            if (line[1] not in lines2):
                patch1.append([line[0], '', line[1]])
                len1 += 1
            else:
                ind = lines2.index(line[1])
                idx2 = lines2_copy.index(line[1])
                patch1.append([line[0], idx2, line[1]])
                lines2.pop(ind)
                l2.pop(ind)
            
        for line in l2:
            patch2.append(['', line[0], line[1]])
            len2 += 1

        i, j = 0, 0
        while (i < len1 and j < len2):
            if (patch1[i][0] < patch2[j][1]):
                patch.append(patch1[i])
                i += 1
            
            elif (patch1[i][0] > patch2[j][1]):
                patch.append(patch2[j])
                j += 1

            else:
                patch.append(patch1[i])
                patch.append(patch2[j])
                i += 1
                j += 1

        for t in range(i, len1):
            patch.append(patch1[t])
        for t in range(j, len2):
            patch.append(patch2[t])
        
        with open('o.txt', 'w', encoding='utf-8') as f:
            for line in patch:
                f.write(str(line[0]).rjust(rjust_num) + ' ' + str(line[1]).rjust(rjust_num)
                         + ' ' + line[2] + '\n')



checker = DiffChecker('out2.txt', '2.txt')
checker.check()