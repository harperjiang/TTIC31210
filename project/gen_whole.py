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


def whole(copy):
    for i in range(copy):
        numWord = random.randint(4, 7)
        numWord2 = random.randint(3, 6)
        numNum = random.randint(5, 8)
        numNum2 = random.randint(4, 6)

        yield "<s> %s - %s - %s - %s </s>\t<s> <WORD> - <WORD> - <NUM> - <NUM> </s>" % (
            breakString(randStr(numWord)), breakString(randStr(numWord2)), breakString(randNum(numNum)),
            breakString(randNum(numNum2)))


# If the group contains more than 3 different types, <WORD> will
# be used. Otherwise, a union will be used

copy = 1000

output = "data/whole.test"

f = open(output, "w")

random.seed(time())

for i in whole(copy):
    f.write("%s\n" % (i))

f.close()
