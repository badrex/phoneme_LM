import LM_models
from collections import defaultdict, Counter
import math

# all Slavic languages which are present in the testing dataset
TGT_LIDs = ['rus', 'ukr', 'pol', 'ces', 'slk', 'bul', 'slv',
    'deu', 'swe', 'fra', 'spa', 'ara', 'zho']

# only for training
TRN_LIDs = ['rus', 'bul', 'pol']

# function to read data from file
def read_data(path_to_file, LangIDs):
    """Given a text file for the data and a set of language IDs,
       return a dictionary as data[LangID] = [[word_1], [word_2], ...]."""

    dataset = defaultdict(list)

    with open(path_to_file) as txtFile:
        # iterate through every line in the text file

        for line in txtFile:
            #process every line in the file
            instance = line.strip().split("\t")

            # get LangID and the transcription of the word
            ID, transcription = instance[0], instance[3]

            # add to the dict only if a target language
            if ID in LangIDs:
                # add beginning of word # and
                # end of word '@' tokens to each transcription, then split
                transcription = '# ' + transcription + ' @'
                transcription = transcription.split()

                #  if the length of the word is only one phoneme, skip and go next
                # this was done to avoid words of only one grapheme
                if len(transcription) < 4: continue

                dataset[ID].append(transcription)

    return dataset

train_data = read_data("pron_data/gold_data_train", TRN_LIDs)
test_data  = read_data("pron_data/gold_data_test",  TGT_LIDs)


# make a phenemic set for every language in the training data
phoneme_seq = defaultdict(list)

# from training dataset
for (LID, T) in train_data.items():
    for trans in T:
        phoneme_seq[LID].extend(ph for ph in trans)

phoneme_set = defaultdict(set)

# build language models and display result
print('Computing cross-lingual phonemic similarity:\n')

for L in phoneme_seq:
    phoneme_set[L] = set(phoneme_seq[L])

print('{:^7} > {:^7}: {:>12}'.format('TRAIN', 'TEST', 'Avg. Surprisal'))

for LID in TRN_LIDs:
    phoneLM = LM_models.Absdisc_Phoneme_LM(train_data[LID], phoneme_set[LID])

    for LID_2 in TGT_LIDs:
        print('{:^7} > {:^7}: {:>14.2f}'.format(LID.upper(), LID_2.upper(), phoneLM.surprisal(test_data[LID_2])))

    print()
