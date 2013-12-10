import operator


class Question:
    def __init__(self, label, choices, answer, score=None):
        self.label = label
        self.choices = choices
        self.answer = answer

        if score is None:
            self.score = lambda result, choice: abs(result - choice)
        else:
            self.score = score

    def to_letter(self, index):
        return chr(index + ord('a'))

    def to_index(self, letter):
        return ord(letter) - ord('a')

    def check(self, result):
        nearest = self.closest(result)

        print(self.label)
        print("  result:  {0}".format(result))
        print("  nearest: {0}".format(self.choice(nearest)))
        print("  answer:  {0}".format(self.choice(self.answer)))

        if nearest == self.answer:
            print("  + CORRECT")
        else:
            print("  - INCORRECT")

    def closest_choice(self, scores, result):
        # returns index, nearest
        index, _ = min(enumerate(scores), key=operator.itemgetter(1))
        return self.to_letter(index)

    def choice(self, letter):
        value = self.choices[self.to_index(letter)]
        return "{0}. {1}".format(letter, value)

    def closest(self, result):
        # ends up with list of distances to zero
        scores = [self.score(result, choice) for choice in self.choices]
        return self.closest_choice(scores, result)

