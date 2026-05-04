import csv
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR

output_filename = os.path.join(DATA_DIR, 'ap_nocm.csv')

lowerconductor = 1;
upperconductor = 10000;
names = [e.cremona_label() for e in CremonaDatabase().iter([
    lowerconductor..upperconductor])];

with open(output_filename, 'w', newline='') as csvfile:
    # Define the column headers
    fieldnames = ['Conductor', 'Rank', 'ap_coefficients']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    cnt = 0;
    for e in names:
        cnt = cnt + 1;
        if (mod(cnt, 100) == 0):
            print(cnt);
           
        E = EllipticCurve(e)
        if E.has_cm():
            continue
        conductor = E.conductor()
        rank = E.rank()
        ap_list = E.aplist(10000)

        # Write each row one by one
        writer.writerow({
            'Conductor': conductor,
            'Rank': rank,
            'ap_coefficients': ap_list
        })
