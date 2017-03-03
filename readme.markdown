Programming assignments for Cal Tech's [Machine Learning](http://work.caltech.edu/telecourse) class by Professor Yaser Abu-Mostafa. My notes for this class are available on [my site](http://blaenkdenum.com/notes/machine-learning/).

The assignments are done in Python with [NumPy](http://www.numpy.org/) and other [SciPy](http://www.scipy.org/) packages.

### Automated Grading

I created a `Question` class in `common.question` that can help grade the assignment. It takes a label (the question itself), a list of possible choices from the assignment, the actual answer (from the answer key) and an optional scoring function for each choice. Its constructor is:

``` python
Question(label, choices, answer, score=None)
```

For example, for question 8 in assignment 2 asks for the in-sample error and can be specified as:

``` python
question8 = Question("8. in sample error",
                     [0, 0.1, 0.3, 0.5, 0.8], 'd')

in_sample_error = experiment()
question8.check(in_sample_error)
```

The output for this is:

```
8. in sample error
  result:  0.506176
  nearest: d. 0.5
  answer:  d. 0.5
  + CORRECT
```

The `Question` class scores every choice in the list based on its distance to the result the experiment yielded. The optional `score` function can be provided to score more sophisticated answers, for example question 9 in assignment 2 asks for the hypothesis that is closest to the one that the experiment returned. This can be expressed as:

``` python
question9 = Question("9. hypothesis",
                     [[-1, -0.5, 0.08, 0.13, 1.50, 1.50],
                      [-1, -0.5, 0.08, 0.13, 1.50, 15.0],
                      [-1, -0.5, 0.08, 0.13, 15.0, 1.50],
                      [-1, -1.5, 0.08, 0.13, 0.05, 0.05],
                      [-1, -0.5, 0.08, 1.50, 0.15, 0.15]], 'a',
                     # polynomial scoring lambda
                     lambda result, choice:
                     sum([abs(result_coeff - coeff)
                         for result_coeff, coeff
                         in zip(result, choice)]))
```

This has the intended result:

```
9. hypothesis
  result:  [-0.90466334 -0.00694238  0.01946925  0.01881652  1.45797108  1.43164961]
  nearest: a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5]
  answer:  a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5]
  + CORRECT
```
