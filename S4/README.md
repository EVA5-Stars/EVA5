# **EVA5-Assignment-4**

## Group Members:
- Gaurav Makkar
- Mohit Bhandari
- Suman Debnath
- Sriram Iyengar



The goal of this exercise is to start familiarizing oneself with how CNN architectures can be optimized. Not every problem needs a hammer and especially the problem described in this exercise. This is a simple digit classification problem using the MNIST dataset.

The "hammer" referred to in the previous paragrah is referring to the number of parameters that a model has to learn. If you are a teacher, with a finite attention span, the number of students that you can monitor and teach is equivalent to the number parameters in a model. If one increases the number of parameters in a model, the number of teachers needed will have to be increased. And that will increase the cost of your school (as each teacher needs to be paid). Therefore it is advisable to keep the number of parameters optimal.

The other dimension is accuracy of your model. It is natural that one needs as high an accuracy as possible, but in this case, unless we get an accuracy of at least 99.40%, we will not get the home-baked cake that Rohan has promised us. Life is all about choices, and this exercise is all about the choice we have to make to achieve the two goals - meet the accuracy target in a minimal cost. And one more thing (there is always, "one more thing") we have to do this in a fixed time-frame of 20 epochs. Think of each epoch being a guess in the famous "20 questions" game, and each answer helps you get closer to your goal.

The natural question is "what exactly are we learning"? We are learning the value of each parameter in our model. We have been given a random set of values in each parameter, and as we get better at learning the "true value" of each parameter, we get a pat from the teacher (think of that pat as a reward).

Given that problem description, the initial state of the problem had about 20Million parameters that we were asked to reduce to under 20k. If you look at the problem, the images only have digits. All digits can be easily drawn by circles and lines. Sometimes the lines are vertical and sometimes the are tilted - and sometimes tilted all the way to be flat. So, all that we have to learn is to recognize these simple edges - circles and lines. Each of these edges is a feature. And when these features are combined to form a digit, that is another higher level feature. So, we dont have too many features that we have to learn.

We use that intuition to reduce the number of features (also called kernels) in our model. That is why the initial part of our model only attempts to learn 8 features. And then combines them into different ways - to take it to 16 features.

But, so that we can start to see and process larger and larger part of the picture we will need to start zooming out. This is where something like MaxPooling helps us to see the forest for the trees.

So, we learn edges in one layer, combine them in various forms and then zoom out. And we can continue to repeat that combination of learning. We call each such set of operations a Block. And because we were dealing with a simple problem, we have only 3 blocks. One for learning edges, one for learning patterns and one for learning the whole object. But between each of these a smooth transition is needed - and we call that a transition block.

We use a special operation called a 1x1 operator, that helps us select only the best intermediate images that we produce in each block. This helps us to take the most meaningful images forward to the next block. This is similar to students who will be dropped if they dont meet the cut.

Often times there is information overload and it becomes difficult to learn. Our minds reach a point of saturation. To avoid this, we use something like BatchNormalization, to lower the saturation levels that enable us to learn.

But, just so that we dont become a "ratta master" (a person who is not learning, but remembering what the person saw/ read) we need to hide some information from the student, to see if the student can naturally extend the learnt concepts to mentally complete that information. We do this by selectively dropping some neurons from the network through which parameters updates will not propagate.

These are essentially the concepts we use to train the MNIST model with 99.41% accuracy will about 8k parameters.
