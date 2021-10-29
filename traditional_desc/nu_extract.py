import re

# ANF要输入的长度相同，只有为same seq的时候才能使用此特征提取
def ANF(seqs, **kw):
    encodings = []

    for sequence in seqs:
        code = []
        for j in range(len(sequence)):
            sequence = re.sub('U', 'T', sequence)
            code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
        encodings.append(code)
    return encodings

def binary(seqs, **kw):
    encodings = []

    AA = 'ACGT'
    for sequence in seqs:
        code = []
        sequence = re.sub('U', 'T', sequence)
        for aa in sequence:
            if aa == '-':
                code = code + [0, 0, 0, 0]
                continue
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        encodings.append(code)
    return encodings


#    if check_sequences.get_min_sequence_length(fastas) < gap + 2:
#        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
#        return 0
def CKSNAP(seqs, gap=5, **kw):
    AA = 'ACGT'

    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    for sequence in seqs:
        code = []
        sequence = re.sub('U', 'T', sequence)
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

def DNC(seqs, **kw):
    base = 'ACGT'

    encodings = []

    AADict = {}
    for i in range(len(base)):
        AADict[base[i]] = i

    for sequence in seqs:
        code = []
        sequence = re.sub('U', 'T', sequence)
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings



if __name__ == '__main__':
    seqs = ['TGATGCAAGCAGTGACTCCTTCAGCTCTGTGCCGTCTCTATTAGCAACAAACCGCTCGGTTTCGATATCCGGTGCTGTTGCGGTGGATGAATTTGATGACA',
    'CCGTAGACAAAAGTTTTTTCAATCTATGGGTTTTAGTTTCAAAAAACGAGACTTTGATCTTGATCTTGATCTGTTGGGAGACTCTAGTTCCGACCATATTC']
    # Test ANF
    # ANF_encoding = ANF(seqs)
    # print(ANF_encoding[0])
    # print(ANF_encoding[1])

    # Test binary
    # binary_encoding = binary(seqs)
    # print(binary_encoding)

    # Test DNC
    DNC_encoding = DNC(seqs)
    print(DNC_encoding)


