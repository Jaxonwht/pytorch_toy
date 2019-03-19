# This file samples the first 5000 democratic lines and first 5000 republican files into separate files.

FILE_NAME = "emails.train"
D_OUT = "democratic5000.train"
R_OUT = "republican5000.train"
NUMBER_OF_LINES = 5000

d_count = 0
r_count = 0

with open(FILE_NAME) as f:
    with open(D_OUT, 'w') as out1:
        with open(R_OUT, 'w') as out2:
            while r_count < 5000 or d_count < 5000:
                line = f.readline()
                if line.startswith('d') and d_count < NUMBER_OF_LINES:
                    out1.write(line)
                    d_count += 1
                elif line.startswith('r') and r_count < NUMBER_OF_LINES:
                    out2.write(line)
                    r_count += 1
