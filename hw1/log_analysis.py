import re

log_file = "word_avg.log"

entries = []
entry = None

epoch_re = re.compile("Epoch (\d+), training Loss is ([\d\.]+)")
dev_re = re.compile("Dev loss is ([\d\.]+)")
test_re = re.compile("Test accuracy is ([\d\.]+)")

lines = open(log_file, "rb").readlines()
for l in lines:
    line = l.decode("utf-8").strip()
    if line.startswith("Epoch"):
        if entry is not None:
            entries.append(entry)
        entry = {}
        match = epoch_re.match(line)
        entry['index'] = int(match.group(1))
        entry['train_loss'] = float(match.group(2))
    if line.startswith("Dev") and entry is not None:
        match = dev_re.match(line)
        entry['dev_loss'] = float(match.group(1))
    if line.startswith("Test") and entry is not None:
        match = test_re.match(line)
        entry['test_loss'] = float(match.group(1))
entries.append(entry)

for entry in entries:
    print("{} {} {} {}".format(entry['index'], entry['train_loss'], entry['dev_loss'], entry['test_loss']))
