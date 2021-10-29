import zipfile

def load_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            train_label.append(int(recordsplit[-2]))
            train_dataset.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            test_label.append(int(recordsplit[-2]))
            test_dataset.append(content_split[index + 1])
    return train_dataset, train_label, test_dataset, test_label

# def load_fasta(filename, skip_first=False):
#     with open(filename, 'r') as file:
#         content = file.read()
#     content_split = content.split('\n')
#
#     dataset = []
#     label = []
#     train_label = []
#     train_dataset = []
#     test_label = []
#     test_dataset = []
#
#     for index, record in enumerate(content_split):
#         if index % 2 == 1:
#             continue
#         recordsplit = record.split('|')
#         if recordsplit[-1] == 'training':
#             train_label.append(int(recordsplit[-2]))
#             train_dataset.append(content_split[index + 1])
#         if recordsplit[-1] == 'testing':
#             test_label.append(int(recordsplit[-2]))
#             test_dataset.append(content_split[index + 1])
#     return train_dataset, train_label, test_dataset, test_label


if __name__ == '__main__':
    filename = '../data/test.txt'
    with open(filename, 'r') as file:
        content = file.read()
        # print(content)
        # > AT1G22840.1_532 | 1 | training
        # AGATGAGGCTTTTTTACTTTGCTATATTCTTTTGCCAAATAAAATCTCAAACTTTTTTTGTTTATCATCAATTACGTTCTTGGTGGGAATTTGGCTGTAAT
        # > AT1G44000.1_976 | 1 | training
        # GATTCGACATAAGTCTATCTTCCATACCTTATTTACGTTTCTTCTGTGAGACAAAGTTGTACATTCTCCTGTGTTTTTTTTTGCAAATGATGTAGATTTCT

    content_split = content.split('\n')
    # print(content_split)
    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            train_label.append(int(recordsplit[-2]))
            train_dataset.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            test_label.append(recordsplit[-2])
            test_dataset.append(content_split[index + 1])
    print(train_dataset)
    print(train_label)
    # import torch
    # torch.cuda.LongTensor(train_label), torch.cuda.LongTensor(train_dataset)

    # ['>AT1G22840.1_532|1|training',
    #  'AGATGAGGCTTTTTTACTTTGCTATATTCTTTTGCCAAATAAAATCTCAAACTTTTTTTGTTTATCATCAATTACGTTCTTGGTGGGAATTTGGCTGTAAT',
    #  '>AT1G44000.1_976|1|training',
    #  'GATTCGACATAAGTCTATCTTCCATACCTTATTTACGTTTCTTCTGTGAGACAAAGTTGTACATTCTCCTGTGTTTTTTTTTGCAAATGATGTAGATTTCT',
    #  '>AT1G09770.1_2698|1|training',
    #  'ACTGGAGAGGAAGAGGACATAGCCATAGCCATGGAAGCTTCTGCATAAAAACTTGAGTTTTGTATTGCTTACAAGTTTTAAGGAGACGTAGCTTGACTTTG',
    #  '>AT1G09645.1_586|1|training',
    #  'ACAAAGGCCTCATGTTTGTTTGTGTTCGTTTGTCTGAGCATGTAGGTGGAACTTATCACTTATGGGTATTTAAATTTGAAGTATATATATACGCATACTTT',
    #  '>AT1G22850.1_1097|1|training',
    #  'ATGCTATAAAGGATATTGATGATGATGAGAAGAGAGATGCAAAGTAGGAAACAAGCCAGCGATTGGATAATGGTTTTGACTCTCTAGGATTTGTAAAACGC',
    #  '>AT1G74960.2_2112|1|training',
