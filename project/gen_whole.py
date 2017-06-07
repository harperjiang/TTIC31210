import random
from time import time


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


def wordNum(copy):
    for i in range(copy):
        numWord = random.randint(4, 7)
        numNum = random.randint(5, 8)

        yield "<s> %s </s>\t<s> <WORD> <NUM> </s>" % (breakString(randStr(numWord) + randNum(numNum)))


def wordDashNum(copy):
    for i in range(copy):
        numWord = random.randint(4, 7)
        numNum = random.randint(5, 8)
        yield "<s> %s - %s </s>\t<s> <WORD> - <NUM> </s>" % (
            breakString(randStr(numWord)), breakString(randNum(numNum)))


def wordDashWord(copy):
    for i in range(copy):
        numWord = random.randint(4, 7)
        numWord2 = random.randint(5, 8)

        yield "<s> %s - %s </s>\t<s> <WORD> - <WORD> </s>" % (
            breakString(randStr(numWord)), breakString(randStr(numWord2)))


def numDashNum(copy):
    for i in range(copy):
        numNum = random.randint(4, 7)
        numNum2 = random.randint(5, 8)
        yield "<s> %s - %s </s>\t<s> <NUM> - <NUM> </s>" % (breakString(randNum(numNum)), breakString(randNum(numNum2)))


# If the group contains more than 3 different types, <WORD> will
# be used. Otherwise, a union will be used

copy = 10000
r1 = 0.3
r2 = 0.7

output = "data/part.train"

f = open(output, "w")

random.seed(time())

for i in wordNum(copy):
    f.write("%s\n" % (i))

for i in wordDashNum(copy):
    f.write("%s\n" % (i))

for i in wordDashWord(copy):
    f.write("%s\n" % (i))

for i in numDashNum(copy):
    f.write("%s\n" % (i))
f.close()
