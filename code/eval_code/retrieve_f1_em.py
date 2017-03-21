import csv

epochs = []
cur_epoch=None
with open("log.txt") as log_in:
    for line in log_in:
        words = line.split(" ")
        words = [w.strip(",;") for w in words]
        if line.startswith("Epoch"):
            if cur_epoch:
                epochs.append(cur_epoch)
            cur_epoch = [int(words[1]) ]
        elif line.startswith("F1"):
            cur_epoch.extend([float(words[1]), float(words[3])])
        elif line.startswith("Preci"):
            cur_epoch.extend([float(words[1]), float(words[3]), int(words[4])])

with open("res.csv", "wb") as res_csv:
    writer = csv.writer(res_csv)
    for e in epochs:
        writer.writerow(e)
