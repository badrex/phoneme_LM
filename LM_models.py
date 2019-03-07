"""
A module for phoneme languages models (LMs).
This code contains elegant and well-tested implementations for two bigram
language models as classes:
    [1] Conventional count-based LM that uses additive smoothing (add-one).
    [2] Back-off LM with absolute discounting as smoothing technique.

The implementation also support computing perplexity and surprisal.
"""

from collections import defaultdict, Counter
import math


class Phoneme_LM:
    """A class to represent a bigram language model with add-one smoothing."""

    def __init__(self, corpus, phoneme_set, unk=True):
        """"Make a phoneme language model, given a vocab set and corpus of
            transcribed words in standard IPA."""

        # make counts for phonemic data
        ph_Ugrams = list()   # unigrams
        ph_Bgrams = list()   # bigrams

        for ipa_trans in corpus:
            # extract phoneme unigrams and bigrams
            Ugrams = [ipa_trans[i - 1] for i in range(1, len(ipa_trans))]
            Bgrams = [(ipa_trans[i - 1], ipa_trans[i]) for i in range(1, len(ipa_trans))]

            ph_Ugrams.extend(Ugrams)
            ph_Bgrams.extend(Bgrams)

        # use Counter to get counts for the LM
        self.UCOUNTS = Counter(ph_Ugrams)    # unigram counts
        self.BCOUNTS = Counter(ph_Bgrams)    # bigram counts

        # get length of the phonemic inventory
        self.phoneme_set = phoneme_set

        if unk:   # in case we expect unknown phonemes in the test data
            self.phoneme_set.add('$')    # $ is the <UNK> symbol

        self.estimate_probs()


    def estimate_probs(self):
        """Estimate probabilities for phoneme LM."""
        # main dictionary for prob values
        self.P = defaultdict(defaultdict)
        L = len(self.phoneme_set)

        for ph1 in self.phoneme_set:
            # if the phoneme ph1 has zero count, consider it zero-valued
            Nu = self.UCOUNTS[ph1] if ph1 in self.UCOUNTS else 0    # unigram history count

            for ph2 in self.phoneme_set:
                # if the phoneme bigram (ph1, ph2) has zero count, consider it zero-valued
                Nb = self.BCOUNTS[(ph1, ph2)] if (ph1, ph2) in self.BCOUNTS else 0

                # estimate prob for P(ph2 | ph1) with add 1 as a smoothing technique
                self.P[ph1][ph2] = (Nb + 1)/(Nu + L)


    def logP(self, ph1, ph2):
        """Return base-2 log probablity given two phonemes; current [ph2] and condition [ph1]."""
        return math.log(self.P[ph1][ph2], 2)


    def perplexity(self, test_corpus):
        """Compute the perplexity of a test corpus."""

        logProbs = 0
        M = 0

        for word in test_corpus:
            for i in range(1, len(word)):
                ph_h, ph_i = word[i - 1], word[i]

                # check if test phoneme is out-of-set
                if ph_h not in self.phoneme_set:   ph_h = '$'
                if ph_i not in self.phoneme_set:   ph_i = '$'

                logProbs += self.logP(ph_h, ph_i)
                M += 1

        PP = math.pow(2, ((-1/M)*logProbs))
        return PP


    def surprisal(self, test_corpus):
        """Compute the average surprisal of a test corpus."""
        return math.log(self.perplexity(test_corpus), 2)


    def test(self, verbose=False):
        """Test whether or not the probability mass sums up to one."""

        if verbose:
            print('Size of the phoneme set:', len(self.phoneme_set))

        for ph1 in self.phoneme_set:
            P_sum = sum(self.P[ph1][ph2] for ph2 in self.phoneme_set)
            precision = 10**-10

            assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'

            if verbose:
                print('Test passed!! Probability mass for conidition {:^3} '.format(ph1)  +
                    'sums up to one ~ {:<20}'.format(P_sum))

        print('TEST SUCCESSFUL!')


class Absdisc_Phoneme_LM(Phoneme_LM):
  """A class to represent a back-iff bigram language model,
     with absolute discounting as smoothing technique."""

  def __init__(self, corpus, phoneme_set, unk=True, d=0.5):
      """"Make a phoneme language model, given a vocab set and corpus of transcribed words."""
      self.d = d
      Phoneme_LM.__init__(self, corpus, phoneme_set, unk=True)
      self.estimate_probs()


  def estimate_probs(self):
      """Estimate probabilities for phoneme LM."""
      # main dictionary for prob values
      self.P = defaultdict(defaultdict)
      L = len(self.phoneme_set)

      for ph1 in self.phoneme_set:
          # if the phoneme ph1 has zero count, consider it zero-valued
          Nh = self.UCOUNTS[ph1] if ph1 in self.UCOUNTS else 0    # unigram history count

          if Nh == 0:
              # estimate prob only using unigrams
              # compute the unigram prob
              for ph2 in self.phoneme_set:
                  Ni = self.UCOUNTS[ph2] if ph2 in self.UCOUNTS else 0
                  N = sum(self.UCOUNTS.values())    # RECHECK THIS! NOT SURE!!!
                  nPlus_i = sum(1 for p in self.UCOUNTS if self.UCOUNTS[p] > 0)
                  Zi = (self.d/N)*nPlus_i
                  uniP = max(Ni - self.d, 0) / N + (Zi * (1/L))

                  self.P[ph1][ph2] = uniP

          else:
              # compute normalization factor
              nPlus_h = sum(1 for (p1, p2) in self.BCOUNTS if p1 == ph1)
              Zh = (self.d/Nh)*nPlus_h

              for ph2 in self.phoneme_set:
                  # if the phoneme bigram (ph1, ph2) has zero count, consider it zero-valued
                  Nb = self.BCOUNTS[(ph1, ph2)] if (ph1, ph2) in self.BCOUNTS else 0

                  # compute the unigram prob
                  Ni = self.UCOUNTS[ph2] if ph2 in self.UCOUNTS else 0
                  N = sum(self.UCOUNTS.values())    # RECHECK THIS! NOT SURE!!!
                  nPlus_i = sum(1 for p in self.UCOUNTS if self.UCOUNTS[p] > 0)
                  Zi = (self.d/N)*nPlus_i
                  uniP = max(Ni - self.d, 0) / N + (Zi * (1/L))

                  # estimate prob for P(ph2 | ph1) with add 1 as a smoothing technique
                  self.P[ph1][ph2] =  max(Nb - self.d, 0) / Nh + (Zh * uniP)
