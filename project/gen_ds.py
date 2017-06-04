import random
from time import time

# The dataset generated here consists of n lines and a pattern
# Each n + 1 lines forms a sample

num_sample = 10


def randStr(len):
    string = ""
    for i in range(len):
        string += str(chr(ord('A') + random.randint(0, 25)))
    return string


def randNum(len):
    string = ""
    for i in range(len):
        string += str(random.randint(0, 9))
    return string


def breakString(str):
    return " ".join([c for c in str])


# If the group contains more than 3 different types, <WORD> will
# be used. Otherwise, a union will be used

copy = 1000
r1 = 0.3
r2 = 0.7

output = "data/ml.test"

f = open(output, "w")

random.seed(time())

for i in range(copy):
    r = random.random()
    if r < r1:
        # Generate Single Word
        prefix = randStr(3)
        # Pattern
        f.write("%s\n" % (breakString(prefix) + " <NUM>"))
        for j in range(num_sample):
            suffix = randNum(6)
            f.write("%s\n" % (breakString(prefix + suffix)))
    elif r < r2:
        # Generate Union
        prefix1 = randStr(3)
        prefix2 = randStr(3)

        f.write("<U> %s <SEP> %s </U> <NUM>\n" % (breakString(prefix1), breakString(prefix2)))

        for j in range(num_sample):
            suffix = randNum(6)
            if random.random() >= 0.5:
                f.write("%s\n" % (breakString(prefix1 + suffix)))
            else:
                f.write("%s\n" % (breakString(prefix2 + suffix)))
        pass
    else:
        # Generate <WORD>
        f.write("<WORD> <NUM>\n")
        for j in range(num_sample):
            f.write("%s\n" % (breakString(randStr(3) + randNum(6))))

f.close()
