import operator


class Question:
    def __init__(self, label, choices, answer, test):
        self.label = label
        self.choices = choices
        self.answer = answer
        self.test = test

    def to_letter(self, index):
        return chr(index + ord('a'))

    def to_index(self, letter):
        return ord(letter) - ord('a')

    def check(self, result):
        nearest = self.test(self, result)

        print(self.label)
        print "\tresult: {0}\t nearest: {1}".format(result, self.choice(nearest)),
        print "\tanswer: {0}\t".format(self.choice(self.answer)),

        if nearest == self.answer:
            print("CORRECT")
        else:
            print("INCORRECT")

    def closest_choice(self, scores, result):
        # returns index, nearest
        index, _ = min(enumerate(scores), key=operator.itemgetter(1))
        return self.to_letter(index)

    def choice(self, letter):
        value = self.choices[self.to_index(letter)]
        return "{0}. {1}".format(letter, value)

    def abs_to_zero(self, result):
        # ends up with list of absolute distances to zero
        scores = [abs(result - choice) for choice in self.choices]
        return self.closest_choice(scores, result)

    def closest(self, result):
        scores = [abs(result - choice) for choice in self.choices]
        return self.closest_choice(scores, result)


